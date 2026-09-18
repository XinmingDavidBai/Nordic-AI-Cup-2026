"""Medical appointment ASR + QA solution.

Pipeline:
  1. Transcribe audio with faster-whisper (OpenAI Whisper); keep segment timestamps.
  2. For each question, retrieve the top-k relevant segments (keyword + number overlap).
  3. Ask a local LLM (ollama / llama3.2:3b) to answer yes/no and cite one segment.
     Each question is a separate, small request so the model focuses on one thing.
     llama3.2:3b with format:json produces answers in ~1-2s each (no chain-of-thought).
  4. Map the cited segment index back to audio timestamps.

Hard negatives (wrong dose, wrong drug, wrong duration) are handled by the prompt,
which requires EXACT value matches before answering yes.

Prerequisites
-------------
  pip install faster-whisper
  ollama pull llama3.2:3b    # recommended: ~1.9 GB, ~1-2s per question

Configuration (environment variables)
--------------------------------------
  WHISPER_MODEL_SIZE   "base", "small", "medium"    (default: base)
  OLLAMA_MODEL         "llama3.2:3b", etc.          (default: llama3.2:3b)
  OLLAMA_URL           default http://localhost:11434
  LLM_NUM_PREDICT      token budget per question     (default: 80)
  TOP_K_SEGS           segments retrieved/question   (default: 8)

Timing on a modern machine (Apple Silicon or recent laptop)
-----------------------------------------------------------
  Transcription (base): ~8-12s
  LLM per question:     ~1-2s (llama3.2:3b, format:json)
  10 questions:         ~10-20s
  Total:                ~20-30s   well within the 60s budget
"""

import json
import logging
import math
import os
import re
import tempfile
import time
from typing import Optional

import requests
from faster_whisper import WhisperModel

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'base')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE = os.getenv('WHISPER_COMPUTE', 'int8')

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'nomic-embed-text')
# llama3.2:3b with format:json only needs ~60-80 tokens to output the answer.
LLM_NUM_PREDICT = int(os.getenv('LLM_NUM_PREDICT', '80'))
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '8'))
EMBED_TIMEOUT = int(os.getenv('EMBED_TIMEOUT', '15'))

TOP_K_SEGS = int(os.getenv('TOP_K_SEGS', '8'))

# ------------------------------------------------------------------ #
# Model warm-up (at import time)                                      #
# ------------------------------------------------------------------ #

logger.info('Loading whisper/%s …', WHISPER_MODEL_SIZE)
_WHISPER = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
logger.info('Whisper loaded.')

try:
    _r = requests.post(
        f'{OLLAMA_URL}/api/generate',
        json={
            'model': OLLAMA_MODEL,
            'prompt': 'Say {"answer":false,"segment":null}',
            'stream': False,
            'format': 'json',
            'options': {'num_predict': 20},
        },
        timeout=30,
    )
    logger.info('Ollama warm-up: %s', _r.status_code)
except Exception as _e:
    logger.warning('Ollama warm-up failed (will retry on first request): %s', _e)


# ------------------------------------------------------------------ #
# ASR                                                                 #
# ------------------------------------------------------------------ #

def transcribe(audio_bytes: bytes) -> list[dict]:
    """Return whisper segments, each with start/end/text."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fh:
        fh.write(audio_bytes)
        tmp = fh.name
    try:
        segs, _ = _WHISPER.transcribe(
            tmp,
            language='en',
            vad_filter=True,
            word_timestamps=True,
            vad_parameters={'min_silence_duration_ms': 300},
        )
        return [
            {
                'start': float(s.start), 'end': float(s.end), 'text': s.text.strip(),
                'words': [{'start': float(w.start), 'end': float(w.end), 'word': w.word} for w in (s.words or [])],
            }
            for s in segs
            if s.text.strip()
        ]
    finally:
        os.unlink(tmp)


# ------------------------------------------------------------------ #
# Segment retrieval                                                   #
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', text.lower())


# Medical synonym expansion: words that appear in clinical questions but rarely
# in spoken transcripts are mapped to the conversational equivalents that DO
# appear (e.g. "stethoscope" → ["listen", "chest", "heart", "lungs"]).
# Only the QUERY is expanded; IDF is computed on the raw segment text.
_MEDICAL_SYNONYMS: dict[str, list[str]] = {
    'stethoscope':   ['listen', 'chest', 'heart', 'lungs'],
    'auscultation':  ['listen', 'chest', 'lungs', 'heart', 'sound', 'sounds'],
    'auscultated':   ['listen', 'listened', 'chest', 'lungs', 'heart'],
    'auscultatory':  ['listen', 'chest', 'lungs', 'heart'],
    'unchanged':     ['carry', 'changes', 'continue', 'same', 'maintain'],
    'unaltered':     ['carry', 'changes', 'continue', 'same'],
    'nsaid':         ['ibuprofen', 'ibumetin', 'naproxen', 'aspirin'],
    'ppi':           ['pantoprazole', 'omeprazole', 'proton'],
    'prescribed':    ['prescription', 'prescribe', 'issued', 'written', 'given'],
    'prescription':  ['prescribe', 'prescribed', 'issued', 'written'],
    'referred':      ['referral', 'sent', 'booked', 'specialist'],
    'referral':      ['referred', 'sent', 'booked', 'specialist'],
    'annual':        ['routine', 'follow', 'check', 'regular', 'yearly'],
    'normal':        ['clear', 'unremarkable', 'fine', 'good', 'sound'],
    'abnormal':      ['concern', 'finding', 'issue', 'problem'],
    'examination':   ['examine', 'check', 'found', 'noted', 'looked'],
    'stable':        ['unchanged', 'steady', 'same', 'consistent', 'controlled'],
}


def _expand_query(tokens: list[str]) -> list[str]:
    """Append clinical synonyms for each token that has a known mapping."""
    expanded = list(tokens)
    for t in tokens:
        expanded.extend(_MEDICAL_SYNONYMS.get(t, []))
    return expanded


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Return nomic-embed-text embeddings for a list of texts (one call)."""
    resp = requests.post(
        f'{OLLAMA_URL}/api/embed',
        json={'model': EMBED_MODEL, 'input': texts},
        timeout=EMBED_TIMEOUT,
    )
    return resp.json()['embeddings']


def embed_segments(segments: list[dict]) -> list[list[float]]:
    """Precompute embeddings for all transcript segments (once per conversation)."""
    texts = [f'search_document: {s["text"]}' for s in segments]
    return _embed_batch(texts)


def _build_idf(segments: list[dict]) -> dict[str, float]:
    """IDF over all segments: log((N+1)/(df+1))+1."""
    N = len(segments)
    df: dict[str, int] = {}
    for seg in segments:
        for t in set(_tokenize(seg['text'])):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}


def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    from collections import Counter
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


def retrieve(
    segments: list[dict],
    question: str,
    k: int = TOP_K_SEGS,
    seg_embeddings: Optional[list[list[float]]] = None,
) -> tuple[list[tuple[int, dict]], int]:
    """Return (top-k pairs in temporal order, index of best segment).

    When seg_embeddings is provided (precomputed nomic-embed-text vectors), uses
    semantic cosine similarity so that paraphrases like "listen to chest" match
    "stethoscope" and "no changes" matches "treatment unchanged". Falls back to
    TF-IDF with medical synonym expansion when embeddings are unavailable.
    Numbers in the question always get a bonus to protect exact-value matching.
    """
    q_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', question))

    if seg_embeddings is not None:
        import numpy as np
        q_emb = np.array(_embed_batch([f'search_query: {question}'])[0])
        q_norm = np.linalg.norm(q_emb) or 1.0
        scored: list[tuple[float, int]] = []
        for i, (seg, s_emb) in enumerate(zip(segments, seg_embeddings)):
            s_arr = np.array(s_emb)
            sim = float(np.dot(q_emb, s_arr) / (q_norm * (np.linalg.norm(s_arr) or 1.0)))
            s_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', seg['text']))
            num_bonus = 0.15 * len(q_nums & s_nums)
            scored.append((sim + num_bonus, i))
    else:
        idf = _build_idf(segments)
        q_toks = _tokenize(question)
        q_vec = _tfidf_vec(_expand_query(q_toks), idf)
        scored = []
        for i, seg in enumerate(segments):
            s_toks = _tokenize(seg['text'])
            sim = _cosine(q_vec, _tfidf_vec(s_toks, idf))
            s_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', seg['text']))
            num_bonus = 0.15 * len(q_nums & s_nums)
            scored.append((sim + num_bonus, i))

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_idx = scored[0][1]
    top_idx = sorted(i for _, i in scored[:k])
    return [(i, segments[i]) for i in top_idx], best_idx


def _clauses(words: list[dict], pause: float = 0.6) -> list[tuple[float, float, str]]:
    """Split a segment's words at punctuation or pauses."""
    units: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and w['start'] - cur[-1]['end'] > pause:
            units.append(cur)
            cur = []
        cur.append(w)
        if w['word'].strip().endswith(('.', '?', '!', ',', ';')):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return [(u[0]['start'], u[-1]['end'], ''.join(w['word'] for w in u).strip()) for u in units]


def refine_span(seg: dict, question: str) -> tuple[float, float]:
    """Narrow a segment to the clause(s) most similar to the question.

    Gold spans are often a single clause of a longer whisper segment; picking
    among {whole segment, each clause, adjacent clause pairs} by embedding
    similarity raised oracle-segment tIoU from 0.654 to 0.708 offline.
    """
    import numpy as np

    cl = _clauses(seg.get('words') or [])
    if len(cl) < 2:
        return seg['start'], seg['end']
    cands = [(seg['start'], seg['end'], seg['text'])] + cl
    cands += [(cl[i][0], cl[i + 1][1], cl[i][2] + ' ' + cl[i + 1][2]) for i in range(len(cl) - 1)]
    try:
        embs = _embed_batch([f'search_query: {question}'] + [f'search_document: {c[2]}' for c in cands])
    except Exception as exc:
        logger.warning('Refine embedding failed (%s); using whole segment', exc)
        return seg['start'], seg['end']
    q = np.array(embs[0])
    c = np.array(embs[1:])
    sims = c @ q / (np.linalg.norm(c, axis=1) * np.linalg.norm(q) + 1e-9)
    best = cands[int(np.argmax(sims))]
    return best[0], best[1]


# ------------------------------------------------------------------ #
# QA via local LLM (ollama, one call per question)                   #
# ------------------------------------------------------------------ #

_SYSTEM = (
    'You are a precise medical transcript analyst. '
    'Answer yes/no questions about a doctor-patient conversation. '
    'Output ONLY valid JSON matching the requested schema.'
)

_PROMPT = """\
Transcript segments (index, timestamp, text):
{context}

Question: {question}

Rules:
- YES if the transcript confirms the claim, even if phrased differently. Common medical paraphrases:
  • "listened to your chest / lungs / heart" or "lungs / chest / heart sound clear / normal" → stethoscope used / auscultation normal
  • "chest and heart both sound normal" → lungs normal on auscultation / heart examination without abnormal findings
  • "carry on as you are / continue as before / no changes" → treatment continues unchanged
  • "I'll prescribe / here is a prescription / take [drug]" → prescription issued for [drug]
  • "referred / sent / booked you for" → referral made
  • "annual / routine / follow-up check" → follow-up visit
  • "still hurting / pain not better" → pain persists despite treatment
  • "under cover of / along with / together with" → combined treatment
- Numbers/quantities must match EXACTLY: "100 mg" is NOT "200 mg"; "2 weeks" is NOT "6 weeks". A near-miss on dose, drug, or duration is NO.
- NO if the topic is absent entirely, or a specific numeric/named value differs from what the question claims.
- If YES, give the index of the ONE segment that most directly proves it.
- If NO, "segment" must be null.

Output JSON: {{"answer": true or false, "segment": integer or null}}"""


def ask(
    segments: list[dict],
    question: str,
    seg_embeddings: Optional[list[list[float]]] = None,
) -> tuple[bool, Optional[tuple[float, float]]]:
    """Query the LLM for one question; return (answer, span_or_None)."""
    top, best_idx = retrieve(segments, question, seg_embeddings=seg_embeddings)
    context = '\n'.join(
        f'[{i}] [{s["start"]:.1f}-{s["end"]:.1f}s]: {s["text"]}'
        for i, s in top
    )
    prompt = _PROMPT.format(context=context, question=question)

    try:
        resp = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': OLLAMA_MODEL,
                'system': _SYSTEM,
                'prompt': prompt,
                'stream': False,
                'format': 'json',
                'options': {'temperature': 0, 'num_predict': LLM_NUM_PREDICT},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        data = json.loads(resp.json().get('response', '{}'))
    except Exception as exc:
        logger.warning('LLM call failed (%s); defaulting to False', exc)
        return False, None

    answer = bool(data.get('answer', False))
    if not answer:
        return False, None

    # Prefer LLM-cited segment: the model saw all context and can match semantics
    # that TF-IDF misses (paraphrases, indirect evidence). Fall back to TF-IDF
    # best only when the cited index is out-of-range or absent.
    llm_seg_idx = data.get('segment')
    top_indices = {i for i, _ in top}
    if isinstance(llm_seg_idx, int) and 0 <= llm_seg_idx < len(segments) and llm_seg_idx in top_indices:
        # Facts are often stated twice; annotators tend to mark the first mention.
        center_idx = min(llm_seg_idx, best_idx)
    else:
        center_idx = best_idx

    return True, refine_span(segments[center_idx], question)


# ------------------------------------------------------------------ #
# Main entry point                                                    #
# ------------------------------------------------------------------ #

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Transcribe audio once; answer each question with a separate LLM call."""
    audio_bytes = decode_audio(request.audio_base64)

    t0 = time.perf_counter()
    segments = transcribe(audio_bytes)
    t_asr = time.perf_counter() - t0

    # Precompute semantic embeddings once for all segments.
    t0 = time.perf_counter()
    try:
        seg_embeddings = embed_segments(segments)
    except Exception as exc:
        logger.warning('Embedding failed (%s); falling back to TF-IDF', exc)
        seg_embeddings = None
    t_emb = time.perf_counter() - t0
    logger.info('%s: %d segments, asr %.1fs, embed %.1fs', request.audio_filename, len(segments), t_asr, t_emb)
    t0 = time.perf_counter()

    answers: list[bool] = []
    evidence_start: list[Optional[float]] = []
    evidence_end: list[Optional[float]] = []

    for question in request.questions:
        try:
            answer, span = ask(segments, question, seg_embeddings=seg_embeddings)
        except Exception:
            logger.exception('Error on question: %s', question)
            answer, span = False, None

        answers.append(answer)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    logger.info('%s: %d questions in %.1fs', request.audio_filename, len(request.questions), time.perf_counter() - t0)
    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )
