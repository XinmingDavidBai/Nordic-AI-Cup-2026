"""Compare results files and evaluate the 'earlier of LLM pick / retrieval best' rule."""
import json, os, sys
D = os.path.dirname(__file__)
cache = json.load(open(os.path.join(D, 'words_cache.json')))


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


for name in sys.argv[1:]:
    res = json.load(open(os.path.join(D, name)))
    pos = [r for r in res if r['gold'] is not None]
    for r in pos:
        segs = cache[r['sid']]
        if r['span'] and r.get('chosen_idx') is None:
            r['chosen_idx'] = next((i for i, s in enumerate(segs) if s['start'] - 1e-3 <= r['span'][0] and r['span'][1] <= s['end'] + 1e-3), None)
    yes = [r for r in pos if r['said'] and r['chosen_idx'] is not None]
    acc = sum(r['said'] == r['want'] for r in res) / len(res)
    mt = sum(r['iou'] for r in pos) / len(pos)
    fn = sum(not r['said'] for r in pos)
    fp = sum(r['said'] for r in res if r['gold'] is None)
    right = sum(r['chosen_idx'] == r['oracle_idx'] for r in yes)
    earlier = sum(min(r['chosen_idx'], r['best_idx']) == r['oracle_idx'] for r in yes)
    print(f'{name}: acc {acc:.3f} tIoU {mt:.3f} score {0.4*acc+0.6*mt:.3f} | FN {fn} FP {fp} | pick right {right}/{len(yes)}, earlier-rule right {earlier}/{len(yes)}')
