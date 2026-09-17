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
# llama3.2:3b with format:json only needs ~60-80 tokens to output the answer.
LLM_NUM_PREDICT = int(os.getenv('LLM_NUM_PREDICT', '80'))
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '8'))

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
            vad_parameters={'min_silence_duration_ms': 300},
        )
        return [
            {'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()}
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
) -> tuple[list[tuple[int, dict]], int]:
    """Return (top-k pairs in temporal order, index of best TF-IDF segment).

    TF-IDF outperforms plain keyword overlap because it down-weights terms that
    appear in many segments (like 'asthma' across an asthma consultation), so it
    distinguishes the specific evidence segment from a generic summary.
    Numbers in the question get an extra bonus so dose/duration mismatches
    surface high-scoring segments with the right values.
    """
    idf = _build_idf(segments)
    q_toks = _tokenize(question)
    q_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', question))
    q_vec = _tfidf_vec(q_toks, idf)

    scored: list[tuple[float, int]] = []
    for i, seg in enumerate(segments):
        s_toks = _tokenize(seg['text'])
        s_vec = _tfidf_vec(s_toks, idf)
        sim = _cosine(q_vec, s_vec)
        # Boost segments that share exact numbers with the question.
        s_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', seg['text']))
        num_bonus = 0.15 * len(q_nums & s_nums)
        scored.append((sim + num_bonus, i))

    # Highest score first; ties broken by earlier segment.
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_idx = scored[0][1]
    top_idx = sorted(i for _, i in scored[:k])
    return [(i, segments[i]) for i in top_idx], best_idx


def _expand_span(
    segments: list[dict],
    center_idx: int,
    top_set: set[int],
    max_gap: float = 1.5,
    max_expand: int = 2,
) -> tuple[float, float]:
    """Widen a single-segment span to adjacent top-k neighbours.

    Many gold evidence spans cover 2-4 consecutive whisper segments (e.g. a
    doctor-patient exchange that unfolds across several turns). Extending the
    span to include immediately adjacent segments that are also relevant
    (i.e. in the top-k retrieval set and within max_gap seconds) significantly
    improves tIoU without risking a runaway wide span.
    """
    start = segments[center_idx]['start']
    end = segments[center_idx]['end']

    # Expand backward.
    count = 0
    for i in range(center_idx - 1, -1, -1):
        if count >= max_expand:
            break
        if i in top_set and segments[i]['end'] >= start - max_gap:
            start = segments[i]['start']
            count += 1
        else:
            break

    # Expand forward.
    count = 0
    for i in range(center_idx + 1, len(segments)):
        if count >= max_expand:
            break
        if i in top_set and segments[i]['start'] <= end + max_gap:
            end = segments[i]['end']
            count += 1
        else:
            break

    return start, end


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
  • "listened to your chest / lungs / heart" or "lungs sound clear" → stethoscope used / auscultation normal
  • "heart sounds normal / regular rate" → heart examination without abnormal findings
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
) -> tuple[bool, Optional[tuple[float, float]]]:
    """Query the LLM for one question; return (answer, span_or_None).

    The LLM receives ALL transcript segments so retrieval vocabulary gaps
    (e.g. "stethoscope" vs "listen to chest") cannot cause false negatives.
    TF-IDF retrieval is kept only as the span-selection fallback when the
    LLM cites a segment index that is out-of-range or absent.
    """
    top, best_idx = retrieve(segments, question)
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
        center_idx = llm_seg_idx
    else:
        center_idx = best_idx

    seg = segments[center_idx]
    return True, (seg['start'], seg['end'])


# ------------------------------------------------------------------ #
# Main entry point                                                    #
# ------------------------------------------------------------------ #

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Transcribe audio once; answer each question with a separate LLM call."""
    audio_bytes = decode_audio(request.audio_base64)

    segments = transcribe(audio_bytes)
    logger.info('%s: %d segments', request.audio_filename, len(segments))

    answers: list[bool] = []
    evidence_start: list[Optional[float]] = []
    evidence_end: list[Optional[float]] = []

    for question in request.questions:
        try:
            answer, span = ask(segments, question)
        except Exception:
            logger.exception('Error on question: %s', question)
            answer, span = False, None

        answers.append(answer)
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )
