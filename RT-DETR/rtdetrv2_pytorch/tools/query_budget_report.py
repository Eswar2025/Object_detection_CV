"""Build the Group 38 report and an optional stock/adaptive upload comparison."""

import argparse
import colorsys
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.query_budget_eval import (
    MODES, PROJECT_ROOT, coco_predictions, load_detector, load_rgb, predict_one, set_mode, warm_modes,
)
from src.data.dataset.coco_dataset import mscoco_category2name


MODE_LABELS = dict(zip(MODES, ('Stock 300', 'Fixed 200', 'Fixed 100', 'Adaptive')))
MODE_COLORS = ('#40576f', '#965aa4', '#07817e', '#c25130')
DRAW_SCORE = 0.35
MATCH_IOU = 0.50
PANEL_WIDTH, PANEL_HEIGHT = 660, 475


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap = width * height
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def match_boxes(predictions, references):
    """Match score-ordered detections to distinct, same-class references at IoU >= 0.50."""
    remaining = dict(enumerate(references))
    matches = {}
    for prediction in sorted(predictions, key=lambda item: -item['score']):
        eligible = ((iou(prediction['bbox'], ref['bbox']), index) for index, ref in remaining.items()
                    if ref['category_id'] == prediction['category_id'])
        overlap, index = max(eligible, default=(0.0, -1))
        if overlap >= MATCH_IOU:
            matches[index] = overlap
            del remaining[index]
    return matches


def group_detections(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['score'] >= DRAW_SCORE:
            grouped[row['image_id']].append(row)
    return grouped


def compare_examples(test_ids, objects, stock, adaptive, budgets):
    """Pick distinct examples and classify the failure panel from one-to-one matches."""
    agreements = {}
    losses, declines, shared_misses = [], [], []
    for image_id in test_ids:
        references = objects[image_id]
        stock_matches = match_boxes(stock[image_id], references)
        adaptive_matches = match_boxes(adaptive[image_id], references)
        pairs = match_boxes(stock[image_id], adaptive[image_id])
        union = len(stock[image_id]) + len(adaptive[image_id]) - len(pairs)
        agreements[image_id] = sum(pairs.values()) / union if union else 1.0
        misses = stock_matches.keys() - adaptive_matches.keys()
        for index, _ in enumerate(references):
            if index in misses:
                losses.append((len(misses), stock_matches[index], image_id, index))
            elif index in stock_matches and index in adaptive_matches:
                change = stock_matches[index] - adaptive_matches[index]
                if change > 0:
                    declines.append((change, image_id, index))
            elif index not in stock_matches and index not in adaptive_matches:
                shared_misses.append((1 - agreements[image_id], image_id, index))

    if losses:
        _, overlap, focus_id, index = max(losses)
        focus_title = 'Stock detection missed by adaptive'
        focus_detail = f'Stock matched at IoU {overlap:.2f}. Adaptive had no qualifying match.'
    elif declines:
        change, focus_id, index = max(declines)
        focus_title = 'Largest localization decline'
        focus_detail = f'Both modes matched the object. Adaptive IoU dropped by {change:.3g}.'
    elif shared_misses:
        _, focus_id, index = max(shared_misses)
        focus_title = 'Shared miss'
        focus_detail = 'Both modes missed the highlighted ground-truth object.'
    else:
        focus_id = min(test_ids, key=lambda image_id: agreements[image_id])
        index = 0 if objects[focus_id] else None
        focus_title = 'Lowest detection agreement'
        focus_detail = 'The matching check found zero misses or localization declines.'

    remaining = [image_id for image_id in test_ids if image_id != focus_id]
    selected = []

    def take(image_id, slug, title, detail):
        remaining.remove(image_id)
        references = objects[image_id]
        largest = max(references, key=lambda ref: ref['area'], default=None)
        selected.append((image_id, slug, title, detail, largest))

    # Reserve a K=200 image for its own panel before choosing the other scenes.
    middle = next((image_id for image_id in remaining if budgets[image_id] == 200), None)
    if middle is not None:
        take(middle, 'intermediate', 'Intermediate budget', 'Adaptive selected 200 queries.')
    if remaining:
        sparse = min(remaining, key=lambda image_id: (not bool(objects[image_id]), len(objects[image_id])))
        take(sparse, 'sparse', 'Sparse scene', f'{len(objects[sparse])} non-crowd annotations.')
    if remaining:
        busy = max(remaining, key=lambda image_id: len(objects[image_id]))
        take(busy, 'busy', 'Busy scene', f'{len(objects[busy])} non-crowd annotations.')
    if remaining:
        closest = max(remaining, key=lambda image_id: (agreements[image_id], bool(stock[image_id])))
        take(closest, 'agreement', 'Closest available agreement',
             f'Detection agreement {agreements[closest]:.2f} on a 0–1 scale.')
    if middle is None and remaining:
        extra = next((image_id for image_id in remaining
                      if budgets[image_id] not in {budgets[row[0]] for row in selected} | {budgets[focus_id]}),
                     remaining[0])
        detail = ('The disagreement panel contains the only 200-query image.' if budgets[focus_id] == 200
                  else 'Adaptive used 100 or 300 queries throughout this split.')
        take(extra, 'alternate', 'Additional budget example', detail)

    order = {'sparse': 0, 'busy': 1, 'intermediate': 2, 'alternate': 2, 'agreement': 3}
    selected.sort(key=lambda row: order[row[1]])
    focus = objects[focus_id][index] if index is not None else None
    selected.append((focus_id, 'disagreement', focus_title, focus_detail, focus))
    return selected


def category_color(category_id):
    rgb = colorsys.hsv_to_rgb((category_id * 0.61803398875) % 1, 0.72, 0.96)
    return tuple(round(255 * channel) for channel in rgb)


def render_pair(image, stock, adaptive, names, stock_k, adaptive_k, title, detail, ground_truth=None):
    font_path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    title_font = ImageFont.truetype(str(font_path), 26)
    text_font = ImageFont.truetype(str(font_path), 17)
    label_font = ImageFont.truetype(str(font_path), 14)
    gap, margin, image_top = 16, 16, 120
    canvas = Image.new('RGB', (2 * (PANEL_WIDTH + margin) + gap, image_top + PANEL_HEIGHT + margin), '#121c29')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, font=title_font, fill='white')
    draw.text((margin, 47), detail, font=text_font, fill='#d0dce6')
    if ground_truth is not None:
        name = names[ground_truth['category_id']]
        draw.text((margin, 70), f"Ground truth  {name} (annotation {ground_truth['id']}) in gold",
                  font=text_font, fill='#ffd566')

    scaled = image.copy()
    scaled.thumbnail((PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    sx, sy = scaled.width / image.width, scaled.height / image.height
    for column, mode_name, queries, predictions in (
        (0, 'Stock', stock_k, stock), (1, 'Adaptive', adaptive_k, adaptive),
    ):
        left = margin + column * (PANEL_WIDTH + gap)
        draw.text((left, 96), f'{mode_name}   K = {queries}', font=text_font, fill='white')
        draw.rectangle((left, image_top, left + PANEL_WIDTH - 1, image_top + PANEL_HEIGHT - 1), fill='#080e16')
        dx = left + (PANEL_WIDTH - scaled.width) // 2
        dy = image_top + (PANEL_HEIGHT - scaled.height) // 2
        canvas.paste(scaled, (dx, dy))

        def pixels(box):
            x, y, w, h = box
            return (dx + min(max(x * sx, 0), scaled.width - 1),
                    dy + min(max(y * sy, 0), scaled.height - 1),
                    dx + min(max((x + w) * sx, 0), scaled.width - 1),
                    dy + min(max((y + h) * sy, 0), scaled.height - 1))

        labels = [(pixels(det['bbox']), category_color(det['category_id']),
                   f"{names[det['category_id']]} {det['score']:.2f}")
                  for det in predictions if det['score'] >= DRAW_SCORE]
        for box, color, _ in reversed(labels):
            draw.rectangle(box, outline=color, width=2)
        if ground_truth is not None:
            box = pixels(ground_truth['bbox'])
            draw.rectangle(box, outline='#080e16', width=5)
            draw.rectangle(box, outline='#ffd566', width=2)

        occupied = []
        for box, color, label in labels:
            width = draw.textlength(label, font=label_font) + 6
            x = min(box[0], left + PANEL_WIDTH - width)
            for offset in (0, -20, 20, -40, 40, -60, 60):
                y = min(max(box[1] - 18 + offset, image_top), image_top + PANEL_HEIGHT - 18)
                bounds = Bbox.from_bounds(x, y, width, 18)
                if not any(bounds.overlaps(previous) for previous in occupied):
                    break
            occupied.append(bounds)
            draw.rectangle(bounds.extents.tolist(), fill='#080e16')
            draw.text((x + 3, y), label, font=label_font, fill=color)
    return canvas


def plot_results(run_dir, summary, budget_counts, manifest):
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    points = [(summary[mode]['p50_ms'], summary[mode]['map']) for mode in MODES]
    for point, color in zip(points, MODE_COLORS):
        ax.scatter(*point, color=color, s=65, zorder=3)
    ax.margins(x=0.35, y=0.32)
    ax.grid(alpha=0.2)
    ax.set(title=f"RT-DETRv2-R50 on {len(manifest['test_ids'])} COCO val2017 images",
           xlabel='p50 decoded RGB to CPU detections (ms)', ylabel='COCO subset mAP@[.50:.95] (%)')
    footer = ax.text(0.02, 0.02, f"Two timing passes × {len(manifest['timing_ids'])} images · "
                     f"{manifest['environment']['device_name']} · FP32", transform=ax.transAxes, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [Bbox.from_bounds(x - 7, y - 7, 14, 14) for x, y in ax.transData.transform(points)]
    occupied.append(footer.get_window_extent(renderer))
    offsets = ((10, 10), (10, -22), (-10, 10), (-10, -22))
    alternatives = offsets + ((10, 24), (10, -36), (-10, 24), (-10, -36))
    for mode, point, color, preferred in zip(MODES, points, MODE_COLORS, offsets):
        label = ax.annotate(f"{MODE_LABELS[mode]} · {summary[mode]['fps']:.1f} FPS", point,
                            color=color, xytext=preferred, textcoords='offset points')
        label.set_in_layout(False)
        for dx, dy in (preferred, *alternatives):
            label.set_position((dx, dy))
            label.set_ha('left' if dx > 0 else 'right')
            bounds = label.get_window_extent(renderer).expanded(1.04, 1.12)
            if (ax.bbox.contains(*bounds.min) and ax.bbox.contains(*bounds.max)
                    and not any(bounds.overlaps(previous) for previous in occupied)):
                break
        occupied.append(bounds)
    fig.savefig(run_dir / 'map_vs_latency.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
    queries = (100, 200, 300)
    bars = ax.bar([str(k) for k in queries], [budget_counts[k] for k in queries], color=MODE_COLORS[-1])
    ax.bar_label(bars, padding=4)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(ylim=(0, max(budget_counts.values()) * 1.15 + 1), title='Adaptive queries on the held-out split',
           xlabel='Queries sent to all six decoder layers', ylabel='Test images')
    fig.savefig(run_dir / 'adaptive_budget_counts.png', dpi=160)
    plt.close(fig)


def interpretation(summary):
    stock, adaptive = summary['stock_300'], summary['adaptive']
    nearest_mode = min(('fixed_100', 'fixed_200'), key=lambda mode: abs(summary[mode]['mean_k'] - adaptive['mean_k']))
    fixed = summary[nearest_mode]
    gain = stock['p50_ms'] - adaptive['p50_ms']
    gains = [stock[f'pass_{p}_p50_ms'] - adaptive[f'pass_{p}_p50_ms'] for p in (1, 2)]
    spread = max(stock['p50_pass_spread_ms'], adaptive['p50_pass_spread_ms'], abs(gains[0] - gains[1]))
    query_reduction = 100 * (300 - adaptive['mean_k']) / 300
    lines = [f"Adaptive reduced mean queries by {query_reduction:.1f}% and changed subset mAP by "
             f"{adaptive['delta_map']:+.2f} points. Stock minus adaptive p50 was {gain:+.2f} ms "
             f"overall and {gains[0]:+.2f} / {gains[1]:+.2f} ms by pass."]
    if min(gains) > 0 and gain > spread and adaptive['delta_map'] >= -0.5:
        lines.append('Adaptive meets the plan’s practical speed improvement criterion.')
    elif max(gains) > 0 and (min(gains) <= 0 or spread >= gain):
        lines.append(f'Latency is inconclusive with {spread:.2f} ms of pass-to-pass variation.')
    elif min(gains) > 0 and adaptive['delta_map'] < -0.5:
        lines.append('The mAP loss exceeds the plan’s 0.5-point limit.')
    else:
        lines.append('The repeated timing passes did not show a lower adaptive p50.')
    lines.append(f"{MODE_LABELS[nearest_mode]} is the fixed control closest to mean adaptive K. "
                 f"It measured {fixed['p50_ms']:.2f} ms p50 and {fixed['map']:.2f}% mAP.")
    for mode in ('fixed_200', 'fixed_100'):
        if summary[mode]['p50_ms'] <= adaptive['p50_ms'] and summary[mode]['map'] >= adaptive['map']:
            lines.append(f'{MODE_LABELS[mode]} matches or beats adaptive on both measured axes.')
    return ' '.join(lines)


def make_report(run_dir, images=None, annotations=None):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'manifest.json')
    summary = {row['mode']: {key: float(value) for key, value in row.items() if key != 'mode'}
               for row in read_csv(run_dir / 'summary.csv')}
    budgets = {int(row['image_id']): int(row['queries']) for row in read_csv(run_dir / 'per_image.csv')
               if row['phase'] == 'predictions' and row['mode'] == 'adaptive'}
    test_ids = manifest['test_ids']
    if budgets.keys() != set(test_ids):
        raise ValueError('Adaptive prediction rows must cover the frozen test manifest')
    counts = Counter(budgets.values())
    plot_results(run_dir, summary, counts, manifest)

    coco = COCO(str(annotations or manifest['annotations']))
    objects = {image_id: [obj for obj in coco.imgToAnns[image_id] if not obj.get('iscrowd', 0)]
               for image_id in test_ids}
    stock = group_detections(read_json(run_dir / 'predictions_stock_300.json'))
    adaptive = group_detections(read_json(run_dir / 'predictions_adaptive.json'))
    panels = compare_examples(test_ids, objects, stock, adaptive, budgets)
    (run_dir / 'examples').mkdir(exist_ok=True)
    names = {key: value['name'] for key, value in coco.cats.items()}
    panel_links = []
    for number, (image_id, slug, title, detail, gt) in enumerate(panels, start=1):
        file = f'examples/{number:02d}_{slug}.png'
        image = load_rgb(images or manifest['images'], coco, image_id)
        render_pair(image, stock[image_id], adaptive[image_id], names, 300, budgets[image_id],
                    f'{title} · COCO image {image_id}', detail, gt).save(run_dir / file)
        panel_links.append(f'- [{title}]({file}) · image {image_id} · adaptive K={budgets[image_id]}. {detail}')

    lines = [
        '# Group 38 adaptive query selection', '',
        f"RT-DETRv2-R50, FP32, {manifest['environment']['device_name']}, seed {manifest['seed']}. "
        f"COCO val2017 subset of {len(test_ids)} test images with {len(manifest['calibration_ids'])} disjoint "
        'calibration images. All four modes ran all six decoder layers.', '',
        f"Frozen adaptive thresholds A={manifest['thresholds']['A']:.8f} and B={manifest['thresholds']['B']:.8f}.", '',
        '| Mode | Mean K | mAP | Δ mAP points | AP50 | AR@100 | p50 ms | p95 ms | FPS | Pass 1 / 2 p50 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for mode in MODES:
        row = summary[mode]
        lines.append(f"| {MODE_LABELS[mode]} | {row['mean_k']:.1f} | {row['map']:.2f} | {row['delta_map']:+.2f} | "
                     f"{row['ap50']:.2f} | {row['ar100']:.2f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
                     f"{row['fps']:.1f} | {row['pass_1_p50_ms']:.2f} / {row['pass_2_p50_ms']:.2f} |")
    lines.extend(['', f"Timing uses two opposite-order passes on the same {len(manifest['timing_ids'])} images. "
                  + manifest['timing_boundary'], '', interpretation(summary), '',
                  '## R18 and R50 paired pilot', '',
                  f"Stock 300 minus fixed 100 on the same {len(manifest['pilot_ids'])} calibration images.", '',
                  '| Model | Median paired savings (ms) | Median by pass (ms) |', '|---|---:|---:|'])
    for name in ('R18', 'R50'):
        pilot = manifest['pilot'][name]
        passes = ' / '.join(f'{value:+.2f}' for value in pilot['pass_median_paired_savings_ms'])
        lines.append(f"| {name} | {pilot['median_paired_savings_ms']:+.2f} | {passes} |")
    counts_text = ', '.join(f'{counts[k]} at K={k}' for k in (100, 200, 300))
    unused = ', '.join(str(k) for k in (100, 200, 300) if not counts[k]) or 'none'
    lines.extend(['', '## Plots and examples', '', f'Adaptive selected {counts_text}. Unused budgets — {unused}.', '',
                  '- [mAP versus latency](map_vs_latency.png)', '- [Adaptive budget counts](adaptive_budget_counts.png)',
                  '', 'The panels use a 0.35 drawing threshold and the same class colors. '
                  'The matching check uses same-class, one-to-one IoU ≥ 0.50 on non-crowd objects. '
                  'Detection agreement is summed match IoU divided by the detection union.', '', *panel_links])
    if len(panels) < 5:
        lines.extend(['', f'This run has only {len(panels)} distinct test images for examples.'])
    (run_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return run_dir / 'report.md'


def compare_upload(image, model, postprocessor, device, thresholds):
    if image is None:
        return None, 'Upload an RGB image to compare the detectors.'
    image = image.convert('RGB')
    results = {}
    for mode in ('stock_300', 'adaptive'):
        set_mode(model, mode, thresholds)
        results[mode] = predict_one(model, postprocessor, image, device)
    stock, adaptive = results['stock_300'], results['adaptive']
    text = (f'Stock {stock.queries} queries, {stock.latency_ms:.1f} ms\n'
            f'Adaptive {adaptive.queries} queries, {adaptive.latency_ms:.1f} ms\n'
            'Single-image FP32 pipeline timings. Boxes shown at scores of 0.35 and above.')
    pair = render_pair(image, coco_predictions(stock, 0), coco_predictions(adaptive, 0), mscoco_category2name,
                       stock.queries, adaptive.queries, 'Group 38 query budget comparison',
                       f'Stock {stock.latency_ms:.1f} ms  ·  Adaptive {adaptive.latency_ms:.1f} ms  ·  Display score ≥ 0.35')
    return pair, text


def make_upload_demo(run_dir, weights_dir):
    """Load the R50 checkpoint once and return a callable for a Gradio PIL input."""
    manifest = read_json(Path(run_dir) / 'manifest.json')
    source = manifest['models']['R50']
    device = torch.device(manifest['environment']['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    original_weights = Path(source['weights'])
    weights = Path(weights_dir) / original_weights.name
    if not weights.is_file():
        weights = original_weights
    model, postprocessor = load_detector(PROJECT_ROOT / source['config'], weights, device)
    thresholds = tuple(manifest['thresholds'][name] for name in ('A', 'B'))
    warm_modes(model, postprocessor, Image.new('RGB', (640, 640)), device, ('stock_300', 'adaptive'),
               manifest['warmup_forwards_per_mode'], thresholds)
    lock = Lock()

    def compare(image):
        with lock:
            return compare_upload(image, model, postprocessor, device, thresholds)

    return compare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Folder written by query_budget_eval.py')
    parser.add_argument('--images', type=Path, help='Override the image location saved in the manifest')
    parser.add_argument('--annotations', type=Path, help='Override the annotations location in the manifest')
    args = parser.parse_args()
    make_report(args.run, args.images, args.annotations)
