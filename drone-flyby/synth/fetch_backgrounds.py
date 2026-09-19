"""Download varied aerial backgrounds from OpenAerialMap (CC-BY 4.0).

    python -m synth.fetch_backgrounds                    # ~70 locations across Europe
    python -m synth.fetch_backgrounds --count 120 --bbox -10,35,40,71
    python -m synth.fetch_backgrounds --pin              # record the current set in synth/backgrounds_pinned.json (git)
    python -m synth.fetch_backgrounds --pinned           # fresh clone: re-download exactly that set

Each location becomes one mosaic of map tiles at roughly 0.12-0.3 m per pixel
(about the scale of the drone's source frames), cropped to the largest block of
tiles that has imagery everywhere:

    datasets/synth_assets/backgrounds/<oam_id>.jpg
    datasets/synth_assets/backgrounds/backgrounds.json   attribution, scale, location, split

About 15 % of locations are marked split='val' (by a hash of the id, so the
split is stable): the compositor only uses those for its held-out validation
images, so validation measures a place the model never trained on.

Imagery © OpenAerialMap contributors, CC-BY 4.0; each image's provider and
title are kept in backgrounds.json for attribution.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'datasets' / 'synth_assets' / 'backgrounds'
# The images are too big for git (~130 MB); this small list of ids, splits and hashes is committed instead.
PINNED = ROOT / 'synth' / 'backgrounds_pinned.json'
API = 'https://api.openaerialmap.org/meta'
# Never fetch ground from here: the recorded validation run looks like Danish
# suburbs and harbour, and public imagery of the same place would be the
# validation scene by the back door. Informed by the recording, not taken from it.
EXCLUDE_BBOX = '8.0,54.5,13.3,57.8'
_SKIP_TITLE = ('dem', 'dsm', 'dtm', 'ndvi', 'thermal', 'elevation', 'height', 'multispectral', 'nir')


def search(bbox, gsd_from, gsd_to, pages):
    results = []
    for page in range(1, pages + 1):
        r = requests.get(API, params={'bbox': bbox, 'gsd_from': gsd_from, 'gsd_to': gsd_to, 'has_tiled': 'true',
                                      'limit': 200, 'page': page}, timeout=60)
        r.raise_for_status()
        batch = r.json()['results']
        results += batch
        if len(batch) < 200:
            break
    return results


def meters_per_pixel(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** z


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(session, url):
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200 or not r.content:
            return None
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        if (img[..., 3] < 250).mean() > 0.02:      # transparent = outside the imagery
            return None
        img = img[..., :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ((gray < 4) | (gray > 251)).mean() > 0.05 or gray.std() < 4:   # nodata fill or blank
        return None
    return img


def largest_valid_block(valid):
    """(r0, r1, c0, c1) of the largest all-valid rectangle in a small bool grid."""
    rows, cols = valid.shape
    best, best_area = None, 0
    for r0 in range(rows):
        for r1 in range(r0 + 1, rows + 1):
            col_ok = valid[r0:r1].all(axis=0)
            run = start = 0
            for c in range(cols + 1):
                if c < cols and col_ok[c]:
                    if run == 0:
                        start = c
                    run += 1
                    continue
                if run and run * (r1 - r0) > best_area:
                    best, best_area = (r0, r1, start, start + run), run * (r1 - r0)
                run = 0
    return best


def build_mosaic(session, meta, cols, rows, min_cols, min_rows):
    lon0, lat0, lon1, lat1 = meta['bbox']
    lat, lon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    gsd = float(meta.get('gsd') or 0.2)
    # Coarsest zoom that is still at least as fine as ~0.12-0.3 m/px (never finer than needed).
    z = int(math.floor(math.log2(156543.03392 * math.cos(math.radians(lat)) / max(gsd, 0.12))))
    z = max(15, min(21, z))
    tms = meta['properties']['tms']
    xa, ya = tile_xy(lon0, lat1, z)
    xb, yb = tile_xy(lon1, lat0, z)
    # Stay inside the image footprint, centred on it.
    avail_cols, avail_rows = int(xb) - int(xa) - 1, int(yb) - int(ya) - 1
    cols, rows = min(cols, avail_cols), min(rows, avail_rows)
    if cols < min_cols or rows < min_rows:
        return None, f'footprint only {avail_cols}x{avail_rows} tiles at z{z}'
    cx, cy = tile_xy(lon, lat, z)
    x0, y0 = int(cx) - cols // 2, int(cy) - rows // 2
    urls = [(r, c, tms.replace('{z}', str(z)).replace('{x}', str(x0 + c)).replace('{y}', str(y0 + r)))
            for r in range(rows) for c in range(cols)]
    with ThreadPoolExecutor(8) as pool:
        tiles = list(pool.map(lambda u: (u[0], u[1], fetch_tile(session, u[2])), urls))
    valid = np.zeros((rows, cols), bool)
    grid = {}
    for r, c, img in tiles:
        if img is not None and img.shape[:2] == (256, 256):
            valid[r, c] = True
            grid[(r, c)] = img
    block = largest_valid_block(valid)
    if block is None or (block[1] - block[0]) < min_rows or (block[3] - block[2]) < min_cols:
        return None, f'only {valid.sum()}/{valid.size} tiles have imagery'
    r0, r1, c0, c1 = block
    mosaic = np.vstack([np.hstack([grid[(r, c)] for c in range(c0, c1)]) for r in range(r0, r1)])
    return mosaic, dict(zoom=z, m_per_px=round(meters_per_pixel(lat, z), 4), tiles=f'{c1 - c0}x{r1 - r0}')


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin() -> int:
    """Write the current background set (ids, splits, hashes) to PINNED, small enough for git."""
    index = json.loads((OUT / 'backgrounds.json').read_text(encoding='utf-8'))
    for e in index['images']:
        e['sha256'] = file_sha(OUT / e['file'])
    PINNED.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"pinned {len(index['images'])} backgrounds -> {PINNED}")
    return 0


def fetch_pinned(out: Path, pinned: Path, args) -> int:
    """Re-download exactly the pinned locations (same ids, zoom rule and splits), e.g. on a fresh clone.

    Imagery on OpenAerialMap can change or vanish: each image is checked against the
    pinned size and hash, and a location that is gone is reported, never replaced by
    another one. Images already on disk are kept, so this is safe to re-run."""
    want = json.loads(pinned.read_text(encoding='utf-8'))['images']
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'nordic-ai-cup-drone-flyby-synth/1.0 (research; CC-BY attribution kept)'
    kept, identical, failed = [], 0, []
    for n, e in enumerate(want, 1):
        path = out / e['file']
        if not path.is_file():
            try:
                r = session.get(f"{API}/{e['oam_id']}", timeout=60)
                r.raise_for_status()
                mosaic, info = build_mosaic(session, r.json()['results'], args.cols, args.rows,
                                            args.min_cols, args.min_rows)
            except Exception as error:
                mosaic, info = None, f'{type(error).__name__}: {error}'
            if mosaic is not None and [int(mosaic.shape[1]), int(mosaic.shape[0])] != list(e['size_px']):
                mosaic, info = None, f"got {mosaic.shape[1]}x{mosaic.shape[0]} px, pinned {e['size_px']}"
            if mosaic is None:
                failed.append(f"{e['oam_id']} {e['title'][:30]!r}: {info}")
                print(f'  [{n:3d}/{len(want)}] FAILED {failed[-1]}', flush=True)
                continue
            cv2.imwrite(str(path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  [{n:3d}/{len(want)}] {e['split']:5s} {e['oam_id']} {e['title'][:32]!r}", flush=True)
        identical += file_sha(path) == e.get('sha256')
        kept.append({k: v for k, v in e.items() if k != 'sha256'})
    # Only what is on disk goes into the index the compositor reads.
    (out / 'backgrounds.json').write_text(json.dumps({'images': kept}, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'{len(kept)}/{len(want)} pinned backgrounds in {out}; {identical} byte-identical to the pinned files '
          f'(the rest were re-encoded by another OpenCV build or changed upstream)')
    if failed:
        print(f'{len(failed)} pinned backgrounds could not be fetched; the synthetic set would differ:', file=sys.stderr)
        for line in failed:
            print(f'  {line}', file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bbox', default='-10,36,32,71', help='lon/lat search box (default: Europe)')
    parser.add_argument('--count', type=int, default=70, help='Locations to fetch')
    parser.add_argument('--gsd-from', type=float, default=0.05)
    parser.add_argument('--gsd-to', type=float, default=0.35)
    parser.add_argument('--cols', type=int, default=12)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--min-cols', type=int, default=6)
    parser.add_argument('--min-rows', type=int, default=4)
    parser.add_argument('--val-share', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pages', type=int, default=6)
    parser.add_argument('--exclude-bbox', default=EXCLUDE_BBOX,
                        help='lon/lat box never fetched from (default: Denmark/Oresund, where the recorded '
                             "validation scene appears to be); 'none' to disable")
    parser.add_argument('--prefer', help='Regex on the image title; matching locations are fetched first '
                                         '(e.g. hard negatives: "harbo|marina|port|hamn")')
    parser.add_argument('--pin', action='store_true', help='Record the current set in synth/backgrounds_pinned.json')
    parser.add_argument('--pinned', nargs='?', const=str(PINNED), metavar='JSON',
                        help='Fetch exactly the locations in this list (default synth/backgrounds_pinned.json), no search')
    parser.add_argument('--out', default=str(OUT), help='With --pinned: folder to fetch into')
    args = parser.parse_args()
    if args.pin:
        return pin()
    if args.pinned:
        return fetch_pinned(Path(args.out), Path(args.pinned), args)
    exclude = None if args.exclude_bbox.lower() == 'none' else [float(v) for v in args.exclude_bbox.split(',')]

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / 'backgrounds.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {'images': []}
    have = {e['oam_id'] for e in index['images']}

    found = search(args.bbox, args.gsd_from, args.gsd_to, args.pages)
    candidates, cells = [], set()
    rng = random.Random(args.seed)
    rng.shuffle(found)
    if args.prefer:
        prefer = re.compile(args.prefer, re.I)
        found.sort(key=lambda m: not prefer.search(m.get('title') or ''))   # stable: keeps the shuffle within groups
    for m in found:
        title = (m.get('title') or '').lower()
        if any(k in title for k in _SKIP_TITLE) or not (m.get('properties') or {}).get('tms'):
            continue
        lon0, lat0, lon1, lat1 = m['bbox']
        lat = (lat0 + lat1) / 2
        if exclude and lon1 >= exclude[0] and lon0 <= exclude[2] and lat1 >= exclude[1] and lat0 <= exclude[3]:
            continue
        width_m = (lon1 - lon0) * 111320 * math.cos(math.radians(lat))
        height_m = (lat1 - lat0) * 110574
        if width_m < 350 or height_m < 250:
            continue
        cell = (round((lon0 + lon1) / 2 / 0.05), round(lat / 0.05))   # one image per ~4 km cell
        if cell in cells:
            continue
        cells.add(cell)
        candidates.append(m)
    print(f'{len(found)} OpenAerialMap images found, {len(candidates)} usable distinct locations')

    session = requests.Session()
    session.headers['User-Agent'] = 'nordic-ai-cup-drone-flyby-synth/1.0 (research; CC-BY attribution kept)'
    fetched = 0
    for m in candidates:
        if fetched >= args.count:
            break
        if m['_id'] in have:
            continue
        mosaic, info = build_mosaic(session, m, args.cols, args.rows, args.min_cols, args.min_rows)
        title = m.get('title') or ''
        if mosaic is None:
            print(f"  skip {m['_id']} {title[:30]!r}: {info}")
            continue
        split = 'val' if int(hashlib.md5(m['_id'].encode()).hexdigest(), 16) % 1000 < args.val_share * 1000 else 'train'
        name = f"{m['_id']}.jpg"
        cv2.imwrite(str(OUT / name), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lon0, lat0, lon1, lat1 = m['bbox']
        index['images'].append({
            'file': name, 'oam_id': m['_id'], 'title': title, 'provider': m.get('provider'),
            'acquired': (m.get('acquisition_end') or '')[:10], 'platform': m.get('platform'),
            'license': 'CC-BY 4.0 (OpenAerialMap)', 'lat': round((lat0 + lat1) / 2, 4), 'lon': round((lon0 + lon1) / 2, 4),
            'native_gsd': m.get('gsd'), 'size_px': [int(mosaic.shape[1]), int(mosaic.shape[0])], 'split': split, **info,
        })
        index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding='utf-8')
        fetched += 1
        print(f"  [{fetched:3d}] {split:5s} {m['_id']} {title[:32]!r:36s} z{info['zoom']} {info['m_per_px']} m/px {info['tiles']} tiles", flush=True)
    splits = {s: sum(1 for e in index['images'] if e['split'] == s) for s in ('train', 'val')}
    print(f"{len(index['images'])} backgrounds in {OUT} ({splits})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
