"""Calibrate and evaluate Group 38's query budgets on a seeded COCO val subset."""

import argparse
import ast
import csv
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_label2category
from src.zoo.rtdetr import rtdetrv2_decoder


R50_CONFIG = PROJECT_ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml'
R18_CONFIG = PROJECT_ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
R50_WEIGHTS = 'rtdetrv2_r50vd_6x_coco_ema.pth'
R18_WEIGHTS = 'rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'
BASELINE_REVISION = 'df2a96c8'
BASELINE_DECODER = 'rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py'
MODES = ('stock_300', 'fixed_200', 'fixed_100', 'adaptive')
FIXED_BUDGETS = {'stock_300': 300, 'fixed_200': 200, 'fixed_100': 100}
PILOT_MODES = ('stock_300', 'fixed_100')
PREPROCESS = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
TIMING_BOUNDARY = (
    'Single-image decoded RGB through 640x640 resize, host-to-device transfer, '
    'forward (including budget selection), postprocessing and CPU outputs. '
    'Synchronizes CUDA at both boundaries when using a GPU. Excludes disk reads, drawing, serialization and COCOeval.'
)


@dataclass
class Prediction:
    labels: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    queries: int
    latency_ms: float


def load_detector(config, weights, device, expected_layers=6):
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    model = cfg.model
    model.load_state_dict(state, strict=True)
    decoder = model.decoder
    if (decoder.num_queries != 300 or decoder.num_classes != 80
            or decoder.query_select_method != 'default' or decoder.learn_query_content
            or decoder.num_layers != expected_layers
            or decoder.decoder.eval_idx != expected_layers - 1):
        raise ValueError(f'{config} must use 300 feature queries, 80 classes and all {expected_layers} layers')
    postprocessor = cfg.postprocessor
    if not postprocessor.use_focal_loss or postprocessor.num_top_queries != 300:
        raise ValueError(f'{config} must rank 300 query/class pairs in postprocessing')
    print(f'Strictly loaded {weights} ({expected_layers} decoder layers)', flush=True)
    return model.deploy().eval().float().to(device), postprocessor.deploy().to(device)


def make_split(image_ids, seed=38, calibration=64, test=256):
    ids = sorted(image_ids)
    if calibration < 2 or test < 1 or calibration + test > len(ids):
        raise ValueError('Need at least two calibration IDs and a disjoint, nonempty test split')
    random.Random(seed).shuffle(ids)
    return ids[:calibration], ids[calibration:calibration + test]


def load_rgb(images, coco, image_id):
    with Image.open(Path(images) / coco.imgs[image_id]['file_name']) as image:
        return image.convert('RGB')


def set_mode(model, mode, thresholds=None):
    if mode == 'adaptive' and thresholds is None:
        raise ValueError('Adaptive mode requires calibrated thresholds')
    decoder = model.decoder
    decoder.fixed_query_budget = None if mode in ('stock_300', 'adaptive') else FIXED_BUDGETS[mode]
    decoder.adaptive_query_thresholds = thresholds if mode == 'adaptive' else None


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def predict_one(model, postprocessor, image, device):
    """Time a decoded RGB PIL image through CPU detections using the current mode."""
    synchronize(device)
    start = time.perf_counter()
    pixels = PREPROCESS(image).unsqueeze(0).to(device)
    original_size = torch.tensor([image.size]).to(device)
    outputs = model(pixels)
    labels, boxes, scores = postprocessor(outputs, original_size)
    labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return Prediction(labels, boxes, scores, outputs['pred_logits'].shape[1], elapsed_ms)


def fit_policy(model, postprocessor, load_image, calibration_ids, device):
    set_mode(model, 'stock_300')
    decoder = model.decoder
    decoder.collect_query_rank_scores = True
    rows = []
    try:
        for image_id in calibration_ids:
            predict_one(model, postprocessor, load_image(image_id), device)
            s101, s201 = decoder.query_rank_scores.cpu().tolist()
            rows.append({'image_id': image_id, 's101': s101, 's201': s201})
    finally:
        decoder.collect_query_rank_scores = False
        decoder.query_rank_scores = None

    threshold_a = float(np.percentile([row['s101'] for row in rows], 33))
    upper_s201 = [row['s201'] for row in rows if row['s101'] > threshold_a]
    threshold_b = float(np.median(upper_s201 or [row['s201'] for row in rows]))
    return (threshold_a, threshold_b), rows


def git_output(*args):
    return subprocess.check_output(['git', *args], cwd=PROJECT_ROOT, text=True).strip()


def baseline_decoder_methods():
    source = git_output('show', f'{BASELINE_REVISION}:{BASELINE_DECODER}')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'RTDETRTransformerv2')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_get_decoder_input', '_select_topk')]
    scope = vars(rtdetrv2_decoder).copy()
    exec(compile(ast.Module(body=methods, type_ignores=[]), BASELINE_DECODER, 'exec'), scope)
    return scope['_get_decoder_input'], scope['_select_topk']


def assert_shapes(outputs, queries, batch=1):
    assert outputs['pred_logits'].shape == (batch, queries, 80), outputs['pred_logits'].shape
    assert outputs['pred_boxes'].shape == (batch, queries, 4), outputs['pred_boxes'].shape


@torch.inference_mode()
def check_parity(model, postprocessor, load_image, calibration_ids, device):
    original_input, original_topk = baseline_decoder_methods()
    decoder = model.decoder
    for image_id in calibration_ids[:2]:
        image = load_image(image_id)
        pixels = PREPROCESS(image).unsqueeze(0).to(device)
        original_size = torch.tensor([image.size], device=device)
        set_mode(model, 'stock_300')
        with patch.object(decoder, '_get_decoder_input', MethodType(original_input, decoder)), \
                patch.object(decoder, '_select_topk', MethodType(original_topk, decoder)):
            reference = model(pixels)
        stock = model(pixels)
        set_mode(model, 'adaptive', (-1.0, -1.0))
        full_width = model(pixels)
        stock_detections = postprocessor(stock, original_size)
        for result in (reference, stock, full_width):
            assert_shapes(result, 300)
        for key in ('pred_logits', 'pred_boxes'):
            torch.testing.assert_close(stock[key], reference[key])
            torch.testing.assert_close(full_width[key], stock[key])
        torch.testing.assert_close(postprocessor(full_width, original_size), stock_detections)
        print(f'Upstream stock and full-width adaptive parity passed on image {image_id}', flush=True)
    for mode in ('fixed_200', 'fixed_100'):
        set_mode(model, mode)
        fixed = model(pixels)
        assert_shapes(fixed, FIXED_BUDGETS[mode])
        labels, boxes, scores = postprocessor(fixed, original_size)
        assert labels.shape == scores.shape == (1, 300)
        assert boxes.shape == (1, 300, 4)
    set_mode(model, 'stock_300')


@torch.inference_mode()
def check_branches(device):
    decoder = rtdetrv2_decoder.RTDETRTransformerv2(
        hidden_dim=8, feat_channels=[8], feat_strides=[8], num_levels=1,
        nhead=2, num_layers=1, dim_feedforward=16, num_denoising=4,
    ).to(device).eval()
    features = [torch.zeros(1, 8, 20, 20, device=device)]
    memory, shapes = decoder._get_encoder_input(features)
    candidates = torch.arange(300, device=device, dtype=torch.float32)[None, :, None]
    top_memory = candidates.expand(1, 300, 8)
    top_boxes = candidates.expand(1, 300, 4) / 300

    for s101, s201, expected in ((0.5, 0.25, 100), (0.75, 0.5, 200), (0.875, 0.75, 300)):
        logits = torch.full((1, 300, 80), -12.0, device=device)
        above, first, between, second, below = logits.new_tensor([
            (1 + s101) / 2, s101, (s101 + s201) / 2, s201, s201 / 2,
        ]).logit()
        logits[:, :100, 3] = above
        logits[:, 100, 3] = first
        logits[:, 101:200, 7] = between
        logits[:, 200, 7] = second
        logits[:, 201:, 7] = below
        with patch.object(decoder, '_select_topk', return_value=(top_memory, logits, top_boxes)) as select:
            decoder.collect_query_rank_scores = True
            content, boxes, _, _ = decoder._get_decoder_input(memory, shapes)
            torch.testing.assert_close(content, top_memory)
            torch.testing.assert_close(boxes, top_boxes)
            torch.testing.assert_close(decoder.query_rank_scores, logits.new_tensor([s101, s201]))
            decoder.collect_query_rank_scores = False

            decoder.adaptive_query_thresholds = (0.5, 0.5)
            content, boxes, _, _ = decoder._get_decoder_input(memory, shapes)
            torch.testing.assert_close(content, top_memory[:, :expected])
            torch.testing.assert_close(boxes, top_boxes[:, :expected])
            assert decoder.query_rank_scores is None
            assert_shapes(decoder(features), expected)
            assert select.call_args.args[-1] == 300
            decoder.adaptive_query_thresholds = None

    for budget in (100, 200):
        decoder.fixed_query_budget = budget
        assert_shapes(decoder(features), budget)
        assert decoder.query_rank_scores is None

    decoder.fixed_query_budget = None
    decoder.adaptive_query_thresholds = (1.0, 1.0)
    assert_shapes(decoder([features[0].expand(2, -1, -1, -1)]), 300, batch=2)
    decoder.num_queries = 200
    assert_shapes(decoder(features), 200)
    decoder.num_queries = 300
    decoder.query_select_method = 'one2many'
    assert_shapes(decoder(features), 300)
    decoder.query_select_method = 'default'
    decoder.tgt_embed = torch.nn.Embedding(300, 8).to(device)
    decoder.learn_query_content = True
    assert_shapes(decoder(features), 300)
    decoder.learn_query_content = False
    decoder.train()
    targets = [{'labels': torch.tensor([3], device=device),
                'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device)}]
    outputs = decoder(features, targets)
    assert_shapes(outputs, 300)
    assert outputs['dn_meta']['dn_num_split'] == [8, 300]
    print('Controlled 100/200/300 branches, aligned slices, fixed modes and bypass checks passed', flush=True)


def warm_modes(model, postprocessor, image, device, modes, repeats, thresholds=None):
    for mode in modes:
        set_mode(model, mode, thresholds)
        for _ in range(repeats):
            predict_one(model, postprocessor, image, device)


def coco_predictions(prediction, image_id):
    boxes = prediction.boxes.clone()
    boxes[:, 2:] -= boxes[:, :2]
    return [
        {'image_id': image_id, 'category_id': mscoco_label2category[label], 'bbox': box, 'score': score}
        for label, box, score in zip(prediction.labels.tolist(), boxes.tolist(), prediction.scores.tolist())
    ]


def benchmark(model, postprocessor, load_image, ids, device, modes, phase, pass_number,
              thresholds=None, keep_predictions=False):
    rows = []
    detections = {mode: [] for mode in modes} if keep_predictions else None
    for image_id in ids:
        image = load_image(image_id)
        for position, mode in enumerate(modes, start=1):
            set_mode(model, mode, thresholds)
            prediction = predict_one(model, postprocessor, image, device)
            assert prediction.queries in (100, 200, 300)
            assert prediction.queries == FIXED_BUDGETS.get(mode, prediction.queries)
            rows.append({
                'phase': phase, 'pass': pass_number, 'position': position, 'image_id': image_id,
                'mode': mode, 'queries': prediction.queries, 'latency_ms': prediction.latency_ms,
            })
            if detections is not None:
                detections[mode].extend(coco_predictions(prediction, image_id))
    return rows, detections


def paired_pilot(name, model, postprocessor, load_image, ids, device, warmup):
    warm_modes(model, postprocessor, load_image(ids[0]), device, PILOT_MODES, warmup)
    paired = []
    stock_times = []
    pass_medians = []
    outside_ms = []
    for pass_number in range(1, 4):
        order = PILOT_MODES if pass_number % 2 else PILOT_MODES[::-1]
        start = time.perf_counter()
        rows, _ = benchmark(model, postprocessor, load_image, ids, device, order, 'pilot', pass_number)
        outside_ms.append(max(0.0, (time.perf_counter() - start) * 1000
                              - sum(row['latency_ms'] for row in rows)) / len(rows))
        times = {(row['image_id'], row['mode']): row['latency_ms'] for row in rows}
        differences = []
        for image_id in ids:
            stock = times[image_id, 'stock_300']
            fixed = times[image_id, 'fixed_100']
            paired.append({'model': name, 'pass': pass_number, 'image_id': image_id,
                           'first_mode': order[0], 'stock_300_ms': stock, 'fixed_100_ms': fixed,
                           'paired_savings_ms': stock - fixed})
            stock_times.append(stock)
            differences.append(stock - fixed)
        pass_medians.append(float(np.median(differences)))
    summary = {
        'median_paired_savings_ms': float(np.median([row['paired_savings_ms'] for row in paired])),
        'pass_median_paired_savings_ms': pass_medians,
        'stock_p95_ms': float(np.percentile(stock_times, 95)),
        'mean_outside_pipeline_ms_per_forward': float(np.mean(outside_ms)),
    }
    print(f"{name} paired savings (300 minus 100): {summary['median_paired_savings_ms']:.3f} ms; "
          f"pass medians {[round(value, 3) for value in pass_medians]}", flush=True)
    return paired, summary


def score_coco(coco, detections, ids):
    if {row['image_id'] for row in detections} != set(ids):
        raise ValueError('Prediction image IDs must match the frozen test manifest')
    evaluated = COCOeval(coco, coco.loadRes(detections), 'bbox')
    evaluated.params.imgIds = list(ids)
    evaluated.evaluate()
    evaluated.accumulate()
    evaluated.summarize()
    return dict(zip(('map', 'ap50', 'ar100'), (float(100 * evaluated.stats[i]) for i in (0, 1, 8))))


def summarize_mode(mode, accuracy, prediction_rows, timing_rows):
    queries = [row['queries'] for row in prediction_rows if row['mode'] == mode]
    mode_timings = [row for row in timing_rows if row['mode'] == mode]
    times = [row['latency_ms'] for row in mode_timings]
    p50, p95 = np.percentile(times, (50, 95))
    pass_p50s = [float(np.median([row['latency_ms'] for row in mode_timings if row['pass'] == p]))
                for p in (1, 2)]
    return {
        'mode': mode, 'mean_k': float(np.mean(queries)), **accuracy,
        'p50_ms': float(p50), 'p95_ms': float(p95), 'fps': 1000 * len(times) / sum(times),
        'pass_1_p50_ms': pass_p50s[0], 'pass_2_p50_ms': pass_p50s[1],
        'p50_pass_spread_ms': abs(pass_p50s[0] - pass_p50s[1]),
    }


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=None if isinstance(data, list) else 2) + '\n',
                    encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Directory of COCO val2017 images')
    parser.add_argument('--annotations', type=Path, required=True, help='COCO instances_val2017.json')
    parser.add_argument('--weights-dir', type=Path, default=Path('weights'))
    parser.add_argument('--r50-config', type=Path, default=R50_CONFIG)
    parser.add_argument('--r18-config', type=Path, default=R18_CONFIG)
    parser.add_argument('--r50-weights', type=Path, help='Override the R50 file under --weights-dir')
    parser.add_argument('--r18-weights', type=Path, help='Override the R18 file under --weights-dir')
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--calibration', type=int, default=64)
    parser.add_argument('--test', type=int, default=256)
    parser.add_argument('--pilot-images', type=int, default=24)
    parser.add_argument('--timing-images', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup forwards per mode')
    parser.add_argument('--inference-budget-minutes', type=float, default=13,
                        help='Freeze test at 128 IDs if the pilot projects an overrun (0 disables)')
    parser.add_argument('--device', default='cuda', help='CUDA by default, CPU for smoke checks')
    parser.add_argument('--out', type=Path, default=Path('outputs/group38'))
    parser.add_argument('--check-only', action='store_true', help='Run branch, parity and checkpoint checks')
    args = parser.parse_args()
    if min(args.pilot_images, args.timing_images) < 1 or args.warmup < 0 or args.inference_budget_minutes < 0:
        parser.error('Timing sample sizes must be positive and warmup and time budget must be nonnegative')
    return args


def model_source(config, weights, layers):
    config = config.resolve()
    if PROJECT_ROOT in config.parents:
        config = config.relative_to(PROJECT_ROOT)
    return {'config': config.as_posix(), 'weights': str(weights.resolve()), 'decoder_layers': layers}


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a GPU runtime, or use --device cpu for smoke checks.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    r50_weights = args.weights_dir / (args.r50_weights or R50_WEIGHTS)
    r18_weights = args.weights_dir / (args.r18_weights or R18_WEIGHTS)
    coco = COCO(str(args.annotations))
    calibration_ids, test_ids = make_split(coco.getImgIds(), args.seed, args.calibration, args.test)
    pilot_ids = calibration_ids[:args.pilot_images]

    def load_image(image_id):
        return load_rgb(args.images, coco, image_id)

    check_branches(device)
    r50, post = load_detector(args.r50_config, r50_weights, device)
    check_parity(r50, post, load_image, calibration_ids, device)
    if not args.check_only:
        thresholds, calibration_rows = fit_policy(r50, post, load_image, calibration_ids, device)
        print(f'Frozen R50 thresholds A={thresholds[0]:.8f}, B={thresholds[1]:.8f}', flush=True)
        r50_rows, r50_pilot = paired_pilot('R50', r50, post, load_image, pilot_ids, device, args.warmup)
    del r50, post

    r18, post = load_detector(args.r18_config, r18_weights, device, expected_layers=3)
    if args.check_only:
        result = predict_one(r18, post, load_image(calibration_ids[0]), device)
        assert result.queries == 300 and result.boxes.shape == (300, 4)
        print('R18 checkpoint forward passed', flush=True)
        return
    r18_rows, r18_pilot = paired_pilot('R18', r18, post, load_image, pilot_ids, device, args.warmup)
    del r18, post

    per_forward_ms = r50_pilot['stock_p95_ms'] + r50_pilot['mean_outside_pipeline_ms_per_forward']
    forward_count = len(MODES) * (len(test_ids) + 2 * min(len(test_ids), args.timing_images) + args.warmup)
    projected_minutes = per_forward_ms * forward_count / 60_000
    if args.inference_budget_minutes and projected_minutes > args.inference_budget_minutes and len(test_ids) > 128:
        test_ids = test_ids[:128]
        print(f'R50 pilot projects {projected_minutes:.1f} min. Freezing the test prefix at 128 IDs.', flush=True)
    timing_ids = test_ids[:args.timing_images]
    manifest = {
        'revision': git_output('rev-parse', 'HEAD'), 'parity_reference': BASELINE_REVISION,
        'project_worktree_modified': bool(git_output('status', '--porcelain', '--', 'configs', 'src', 'tools')),
        'seed': args.seed, 'calibration_ids': calibration_ids, 'test_ids': test_ids,
        'pilot_ids': pilot_ids, 'timing_ids': timing_ids, 'requested_test_size': args.test,
        'thresholds': dict(zip(('A', 'B'), thresholds)), 'warmup_forwards_per_mode': args.warmup,
        'timing_pass_orders': [MODES, MODES[::-1]], 'timing_boundary': TIMING_BOUNDARY,
        'timing_projection': {'requested_test_minutes': projected_minutes,
                              'inference_budget_minutes': args.inference_budget_minutes,
                              'method': 'R50 stock p95 plus pilot overhead per forward'},
        'images': str(args.images.resolve()), 'annotations': str(args.annotations.resolve()),
        'models': {'R50': model_source(args.r50_config, r50_weights, 6),
                   'R18': model_source(args.r18_config, r18_weights, 3)},
        'pilot': {'R50': r50_pilot, 'R18': r18_pilot},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                        'cuda': torch.version.cuda, 'device': str(device), 'precision': 'FP32',
                        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'manifest.json', manifest)
    write_csv(args.out / 'calibration.csv', calibration_rows)
    write_csv(args.out / 'pilot.csv', r50_rows + r18_rows)

    r50, post = load_detector(args.r50_config, r50_weights, device)
    warm_modes(r50, post, load_image(test_ids[0]), device, MODES, args.warmup, thresholds)
    print(f'Running all four modes on {len(test_ids)} shared test IDs', flush=True)
    prediction_rows, detections = benchmark(r50, post, load_image, test_ids, device, MODES,
                                             'predictions', 0, thresholds, keep_predictions=True)
    timing_rows = []
    for pass_number, order in enumerate((MODES, MODES[::-1]), start=1):
        print(f'Timing pass {pass_number} on {len(timing_ids)} shared IDs', flush=True)
        rows, _ = benchmark(r50, post, load_image, timing_ids, device, order, 'timing', pass_number, thresholds)
        timing_rows.extend(rows)
    del r50, post

    write_csv(args.out / 'per_image.csv', prediction_rows + timing_rows)
    for mode, predictions in detections.items():
        write_json(args.out / f'predictions_{mode}.json', predictions)
    summary = []
    for mode, predictions in detections.items():
        print(f'COCOeval {mode} on {len(test_ids)} test IDs', flush=True)
        accuracy = score_coco(coco, predictions, test_ids)
        summary.append(summarize_mode(mode, accuracy, prediction_rows, timing_rows))
    for row in summary:
        row['delta_map'] = row['map'] - summary[0]['map']
    write_csv(args.out / 'summary.csv', summary)
    print(f'Saved run to {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
