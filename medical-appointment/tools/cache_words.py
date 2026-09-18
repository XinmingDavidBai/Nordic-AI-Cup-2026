"""Transcribe every sample with word timestamps and cache to JSON."""
import json, os, time
from faster_whisper import WhisperModel

OUT = os.path.join(os.path.dirname(__file__), 'words_cache.json')
model = WhisperModel('base', device='cpu', compute_type='int8')
cache = {}
files = sorted(f for f in os.listdir('data/audio') if f.endswith('.mp3'))
for f in files:
    sid = f.replace('conversation_', '').replace('.mp3', '')
    t0 = time.time()
    segs, _ = model.transcribe(f'data/audio/{f}', language='en', vad_filter=True, word_timestamps=True,
                               vad_parameters={'min_silence_duration_ms': 300})
    cache[sid] = [
        {'start': s.start, 'end': s.end, 'text': s.text.strip(),
         'words': [{'start': w.start, 'end': w.end, 'word': w.word} for w in (s.words or [])]}
        for s in segs if s.text.strip()
    ]
    print(f'{sid}: {len(cache[sid])} segs in {time.time()-t0:.1f}s', flush=True)
json.dump(cache, open(OUT, 'w'))
print('wrote', OUT)
