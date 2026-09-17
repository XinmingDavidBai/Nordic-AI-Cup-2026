"""Post single conversations to the running server and time the round trip."""
import base64, csv, sys, time
import requests

rows = list(csv.DictReader(open('data/question_train.csv')))
for sid in sys.argv[1:]:
    qs = [r['question'] for r in rows if r['transcript_id'] == sid]
    audio = base64.b64encode(open(f'data/audio/conversation_{sid}.mp3', 'rb').read()).decode()
    t0 = time.time()
    r = requests.post('http://localhost:9054/predict', json={
        'audio_base64': audio, 'audio_filename': f'conversation_{sid}.mp3', 'questions': qs}, timeout=120)
    print(f'{sid}: {r.status_code} in {time.time()-t0:.1f}s, yes={sum(r.json()["answers"])}/{len(qs)}', flush=True)
