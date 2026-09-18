import json, os, sys
from collections import Counter
D = os.path.dirname(__file__)
res = json.load(open(os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else 'pipeline_results.json')))
cache = json.load(open(os.path.join(D, 'words_cache.json')))

pos = [r for r in res if r['gold'] is not None]
yes = [r for r in pos if r['said']]
wrong = [r for r in yes if r['chosen_idx'] != r['oracle_idx']]
print(f'wrong picks: {len(wrong)}/{len(yes)}; mean IoU on wrong {sum(r["iou"] for r in wrong)/len(wrong):.3f}, on right {sum(r["iou"] for r in yes if r["chosen_idx"]==r["oracle_idx"])/(len(yes)-len(wrong)):.3f}')
offs = Counter()
for r in wrong:
    d = None if r['chosen_idx'] is None else r['chosen_idx'] - r['oracle_idx']
    offs[d] += 1
print('offset chosen-oracle:', sorted(offs.items(), key=lambda x: (x[0] is None, x[0])))
print('chosen==best_idx among wrong:', sum(r['chosen_idx'] == r['best_idx'] for r in wrong))
print('best_idx==oracle among wrong (LLM overrode a correct retrieval):', sum(r['best_idx'] == r['oracle_idx'] for r in wrong))
print('oracle in topk among wrong:', sum(r['oracle_in_topk'] for r in wrong))

print('\n=== WRONG PICK EXAMPLES ===')
for r in wrong[:25]:
    segs = cache[r['sid']]
    o, c = r['oracle_idx'], r['chosen_idx']
    print(f'\n[{r["qid"]}] {r["question"]}  gold={r["gold"]} iou={r["iou"]:.2f}')
    print(f'   GOLD  [{o}] {segs[o]["start"]:.1f}-{segs[o]["end"]:.1f}: {segs[o]["text"]}')
    if c is not None:
        print(f'   CHOSE [{c}] {segs[c]["start"]:.1f}-{segs[c]["end"]:.1f}: {segs[c]["text"]}')
    else:
        print(f'   CHOSE span={r["span"]}')

print('\n=== FALSE NEGATIVES ===')
for r in pos:
    if not r['said']:
        segs = cache[r['sid']]; o = r['oracle_idx']
        print(f'[{r["qid"]}] {r["question"]}\n   GOLD [{o}] in_topk={r["oracle_in_topk"]}: {segs[o]["text"]}')

print('\n=== FALSE POSITIVES ===')
for r in res:
    if r['gold'] is None and r['said']:
        print(f'[{r["qid"]}] ({r["type"]}) {r["question"]}')
