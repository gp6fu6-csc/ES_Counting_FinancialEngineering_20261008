"""PDF-aligned three-ROI FFT rotation correction."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROI_WIDTH = 1024
ROI_HEIGHT = 512
EXPECTED_IMAGE_SHAPE = (2048, 2448)
QUALITY_TOLERANCE_DEG = 0.25
DC_RADIUS = 5
CANNY_LOW = 50
CANNY_HIGH = 150
NORMALIZED_PEAK_THRESHOLD = 1e-6


def normalize_angle(angle):
    return (angle + 90.0) % 180.0 - 90.0


def roi_boxes(height, width):
    if width < ROI_WIDTH or height < ROI_HEIGHT:
        raise ValueError(f"image {width}x{height} is smaller than ROI {ROI_WIDTH}x{ROI_HEIGHT}")
    x = (width - ROI_WIDTH) // 2
    boxes = []
    for center_y in (height / 6, height / 2, 5 * height / 6):
        y = int(round(center_y - ROI_HEIGHT / 2))
        y = max(0, min(height - ROI_HEIGHT, y))
        boxes.append((x, y, ROI_WIDTH, ROI_HEIGHT))
    return boxes


def estimate_roi(roi):
    data = roi.astype(np.float32) - float(np.mean(roi))
    window = np.outer(np.hanning(roi.shape[0]), np.hanning(roi.shape[1]))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(data * window)))
    cy, cx = (np.array(spectrum.shape) // 2).tolist()
    yy, xx = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 > DC_RADIUS ** 2) & (yy - cy > 5)
    masked = np.where(mask, spectrum, -1.0)
    py, px = np.unravel_index(np.argmax(masked), spectrum.shape)

    def subpixel(axis, index):
        if index <= 0 or index >= len(axis) - 1:
            return 0.0
        left, center, right = [max(float(axis[index + offset]), 1e-12) for offset in (-1, 0, 1)]
        denominator = math.log(left) - 2 * math.log(center) + math.log(right)
        return 0.5 * (math.log(left) - math.log(right)) / denominator if denominator else 0.0

    u, v = float(px - cx) + subpixel(spectrum[py, :], px), float(py - cy) + subpixel(spectrum[:, px], py)
    # FFT bins have different physical scales for the rectangular ROI.
    # Convert (u, v) bins to cycles/pixel before calculating the wave-vector
    # angle; theta is the wave-vector angle relative to the image vertical.
    fx_cycles_per_pixel = u / roi.shape[1]
    fy_cycles_per_pixel = v / roi.shape[0]
    wave_angle = math.degrees(math.atan2(fy_cycles_per_pixel, fx_cycles_per_pixel))
    theta = normalize_angle(wave_angle - 90.0)
    peak_value = float(spectrum[py, px])
    normalized_peak = peak_value / max(float(np.sum(spectrum)), 1e-12)
    return {
        "peak_u": u,
        "peak_v": v,
        "peak_fx_cycles_per_pixel": fx_cycles_per_pixel,
        "peak_fy_cycles_per_pixel": fy_cycles_per_pixel,
        "peak_value": peak_value,
        "normalized_peak_magnitude": normalized_peak,
        "theta_deg": theta,
    }


def estimate_image(gray):
    boxes = roi_boxes(*gray.shape)
    estimates = [estimate_roi(gray[y:y + h, x:x + w]) for x, y, w, h in boxes]
    angles = [item["theta_deg"] for item in estimates]
    return boxes, estimates, float(np.median(angles))


def estimate_boxes(image, boxes):
    estimates = [estimate_roi(image[y:y + h, x:x + w]) for x, y, w, h in boxes]
    angles = [item["theta_deg"] for item in estimates]
    aggregate = dict(estimates[0])
    for key in aggregate:
        if isinstance(aggregate[key], (int, float)):
            aggregate[key] = float(np.median([item[key] for item in estimates]))
    aggregate["theta_deg"] = float(np.median(angles))
    return estimates, aggregate


def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def rotate_mask(mask, angle):
    """Rotate the validity map; output keeps mask appearance (0=valid steel, 255=invalid, incl. outside image)."""
    appearance = np.where(normalize_mask(mask), 0, 255).astype(np.uint8)
    height, width = appearance.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(appearance, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """Read an image from path safely supporting Unicode/non-ASCII paths on Windows."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def write_image(path, image):
    """Write an image safely supporting Unicode/non-ASCII paths on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"unable to encode image: {path}")
    try:
        with open(path, "wb") as f:
            f.write(buf)
    except Exception as e:
        raise IOError(f"unable to write image: {path}: {e}")
    if read_image(path) is None:
        raise IOError(f"written image is unreadable: {path}")


def save_comparison(original, corrected, path, compensation):
    height = 600
    scale = height / original.shape[0]
    panels = []
    for image, label in ((original, "original"), (corrected, "corrected")):
        panel = cv2.resize(image, (int(image.shape[1] * scale), height))
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        panels.append(panel)
    cv2.putText(panels[1], f"compensation {compensation:+.4f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    write_image(path, np.hstack(panels))


def save_fft_diagnostic(gray, boxes, estimates, path):
    canvas = cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for index, ((x, y, w, h), item) in enumerate(zip(boxes, estimates), 1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(canvas, f"ROI {index}: theta={item['theta_deg']:.3f} deg", (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        px, py = int(round(x + w / 2 + item["peak_u"])), int(round(y + h / 2 + item["peak_v"]))
        cv2.drawMarker(canvas, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    write_image(path, canvas)


def save_mask_roi_diagnostic(mask, roi, path):
    normalized = normalize_mask(mask)
    canvas = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if roi is not None:
        x, y, width, height = roi["xywh"]
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 3)
        cv2.putText(canvas, f"ROI {width}x{height}", (x + 10, max(35, y + 35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    write_image(path, canvas)


def _masked_horizontal_accumulation(gray, mask, roi=None):
    normalized = normalize_mask(mask)
    if tuple(gray.shape) != tuple(normalized.shape):
        raise ValueError(f"image shape {tuple(gray.shape)} does not match mask shape {tuple(normalized.shape)}")
    x_start = 0
    y_start = 0
    if roi is not None:
        x_start, y_start, width, height = [int(value) for value in roi["xywh"]]
        gray = gray[y_start:y_start + height, x_start:x_start + width]
        normalized = normalized[y_start:y_start + height, x_start:x_start + width]
    valid = normalized.astype(bool)
    valid_counts = np.count_nonzero(valid, axis=0).astype(np.int64)
    intensity_sum = np.sum(np.where(valid, gray, 0), axis=0, dtype=np.float64)
    intensity_mean = np.divide(
        intensity_sum,
        valid_counts,
        out=np.zeros_like(intensity_sum),
        where=valid_counts > 0,
    )
    return valid_counts, intensity_sum, intensity_mean, x_start


def _write_horizontal_signal_csv(path, valid_counts, intensity_sum, intensity_mean, x_start=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("x", "valid_pixel_count", "intensity_sum", "intensity_mean"))
        writer.writeheader()
        for offset, (count, total, mean) in enumerate(zip(valid_counts, intensity_sum, intensity_mean)):
            writer.writerow({
                "x": x_start + offset,
                "valid_pixel_count": int(count),
                "intensity_sum": float(total),
                "intensity_mean": float(mean),
            })


def save_horizontal_signal_diagnostic(intensity_mean, path):
    width = max(640, min(1600, int(len(intensity_mean))))
    height = 480
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.line(canvas, (40, height - 40), (width - 20, height - 40), (0, 0, 0), 2)
    cv2.line(canvas, (40, 20), (40, height - 40), (0, 0, 0), 2)
    if intensity_mean.size:
        values = cv2.normalize(intensity_mean.astype(np.float32), None, 20, height - 40, cv2.NORM_MINMAX)
        points = []
        for index, value in enumerate(values):
            x = 40 + int(round(index * (width - 60) / max(1, len(values) - 1)))
            points.append((x, height - int(round(float(np.asarray(value).item())))))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (255, 0, 0), 2)
    cv2.putText(canvas, "masked horizontal intensity mean", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    write_image(path, canvas)


def _write_analysis_artifacts(gray, mask, output_dir, roi):
    boxes = roi.get("rois_xywh", [roi["xywh"]])
    signals = []
    normalized_mask = normalize_mask(mask)
    for index, box in enumerate(boxes, 1):
        current_roi = {"xywh": list(box)}
        valid_counts, intensity_sum, intensity_mean, x_start = _masked_horizontal_accumulation(gray, mask, current_roi)
        signal_csv = output_dir / f"masked_horizontal_accumulation_roi{index}.csv"
        signal_png = output_dir / f"masked_horizontal_accumulation_roi{index}.png"
        _write_horizontal_signal_csv(signal_csv, valid_counts, intensity_sum, intensity_mean, x_start)
        save_horizontal_signal_diagnostic(intensity_mean, signal_png)
        x, y, width, height = [int(value) for value in box]
        effective_bounds = {"x": x, "y": y, "width": width, "height": height, "xywh": list(box)}
        signals.append({
            "roi_index": index, "roi_xywh": list(box), "effective_bounds": effective_bounds,
            "valid_pixel_count": int(np.count_nonzero(normalized_mask[y:y + height, x:x + width])),
            "signal_csv": str(signal_csv.resolve()), "signal_png": str(signal_png.resolve()),
        })
    primary = signals[0]
    return {
        "status": "unavailable", "model_status": "not_supplied",
        "reason": "Preprocessing-only scope: no validated N/w/D or KD/MACD/count model is supplied by the project inputs.",
        "method": "masked_horizontal_accumulation", "roi_xywh": primary["roi_xywh"],
        "effective_bounds": primary["effective_bounds"], "roi_count": len(signals),
        "valid_pixel_count": primary["valid_pixel_count"],
        "signal_csv": primary["signal_csv"], "signal_png": primary["signal_png"],
        "signals": signals,
        "columns": ["x", "valid_pixel_count", "intensity_sum", "intensity_mean"],
    }


def _file_signature(path):
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _image_channels(image):
    return 1 if image.ndim == 2 else image.shape[2]


def run_with_parameters(input_path, parameters_path, output_path, metadata_path):
    input_path = Path(input_path).resolve()
    parameters_path = Path(parameters_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    source_dir = input_path.parent
    if output_path.parent == source_dir or metadata_path.parent == source_dir:
        raise ValueError("outputs must not be written beside the source image")

    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    try:
        compensation = float(parameters["compensation_angle_deg"])
        rotation = parameters["rotation"]
        interpolation = rotation["interpolation"]
        border = rotation["border"]
        same_size = rotation["same_size"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("parameters JSON is missing the required rotation settings") from error
    if interpolation != "INTER_LINEAR" or border != "BORDER_REFLECT_101" or same_size is not True:
        raise ValueError("parameters JSON rotation settings do not match rotate_image()")

    original_size, original_hash = _file_signature(input_path)
    original = read_image(input_path, cv2.IMREAD_UNCHANGED)
    if original is None:
        raise ValueError(f"unable to read image: {input_path}")
    corrected = rotate_image(original, compensation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, corrected)
    readable = read_image(output_path, cv2.IMREAD_UNCHANGED)
    if readable is None:
        raise IOError(f"written image is unreadable: {output_path}")
    shape_matches = readable.shape == original.shape
    dtype_matches = readable.dtype == original.dtype
    if not shape_matches or not dtype_matches:
        raise IOError("corrected image shape or dtype does not match input")
    after_size, after_hash = _file_signature(input_path)
    original_unchanged = original_size == after_size and original_hash == after_hash
    if not original_unchanged:
        raise IOError("source image changed during correction")

    metadata = {
        "input_path": str(input_path),
        "parameters_path": str(parameters_path),
        "compensation_angle_deg": compensation,
        "rotation": rotation,
        "output_path": str(output_path),
        "original_file_size_before": original_size,
        "original_file_size_after": after_size,
        "original_sha256_before": original_hash,
        "original_sha256_after": after_hash,
        "image": {
            "width": original.shape[1], "height": original.shape[0],
            "channels": _image_channels(original), "dtype": str(original.dtype),
        },
        "verification": {
            "output_readable": True,
            "shape_matches_input": shape_matches,
            "dtype_matches_input": dtype_matches,
            "original_unchanged": original_unchanged,
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def run(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    original = read_image(input_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"unable to read image: {input_path}")
    boxes, before_rois, estimated = estimate_image(original)
    # Convert FFT bins to cycles/pixel before estimating theta, so the
    # physical image-coordinate correction is the same signed angle.
    # OpenCV's positive angle counteracts the measured theta for this
    # wave-vector convention and the selected lower FFT half-plane.
    compensation = estimated
    corrected = rotate_image(original, compensation)
    _, after_rois, after_median = estimate_image(corrected)
    write_image(output_dir / "original.png", original)
    write_image(output_dir / "corrected.png", corrected)
    edges_before = cv2.Canny(original, CANNY_LOW, CANNY_HIGH)
    edges_after = cv2.Canny(corrected, CANNY_LOW, CANNY_HIGH)
    write_image(output_dir / "edges_before.png", edges_before)
    write_image(output_dir / "edges_after.png", edges_after)
    save_comparison(original, corrected, output_dir / "comparison.png", compensation)
    save_fft_diagnostic(original, boxes, before_rois, output_dir / "fft_diagnostic.png")
    before_angles = [x["theta_deg"] for x in before_rois]
    after_angles = [x["theta_deg"] for x in after_rois]
    result = {
        "status": "ok", "input_path": str(input_path.resolve()), "output_dir": str(output_dir.resolve()),
        "image": {"width": original.shape[1], "height": original.shape[0], "channels": 1, "dtype": str(original.dtype)},
        "roi": {"width_px": ROI_WIDTH, "height_px": ROI_HEIGHT, "boxes_xywh": [list(x) for x in boxes], "vertical_centers": ["H/6", "H/2", "5H/6"]},
        "fft": {"dc_exclusion_radius_px": DC_RADIUS, "search_half_plane": "v > 5", "window": "2D Hann", "rois": before_rois},
        "before_theta_i_deg": before_angles, "after_theta_i_deg": after_angles,
        "estimated_angle_deg": estimated, "angle_definition": "theta = normalize(atan2(fy, fx) in degrees - 90), where fx=peak_u/ROI_width and fy=peak_v/ROI_height cycles/pixel", "compensation_direction": "OpenCV positive angle equal to measured theta", "compensation_angle_deg": compensation,
        "before_abs_median_angle_deg": abs(float(np.median(before_angles))),
        "after_abs_median_angle_deg": abs(float(np.median(after_angles))),
        "angle_improvement_deg": abs(float(np.median(before_angles))) - abs(float(np.median(after_angles))),
        "rotation": {"interpolation": "INTER_LINEAR", "border": "BORDER_REFLECT_101", "same_size": True},
        "edges": {"canny_low": CANNY_LOW, "canny_high": CANNY_HIGH, "before_nonzero": int(np.count_nonzero(edges_before)), "after_nonzero": int(np.count_nonzero(edges_after))},
        "output_files": [str(p.resolve()) for p in sorted(output_dir.iterdir()) if p.is_file()],
    }
    summary = [
        f"estimated_angle_deg={estimated:.9f}", f"compensation_angle_deg={compensation:.9f}",
        f"before_theta_i_deg={before_angles}", f"after_theta_i_deg={after_angles}",
        f"before_abs_median_angle_deg={result['before_abs_median_angle_deg']:.9f}",
        f"after_abs_median_angle_deg={result['after_abs_median_angle_deg']:.9f}",
        f"angle_improvement_deg={result['angle_improvement_deg']:.9f}",
        f"edge_nonzero_before={result['edges']['before_nonzero']}", f"edge_nonzero_after={result['edges']['after_nonzero']}",
    ]
    summary_path = output_dir / "verification_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    if not summary_path.is_file():
        raise IOError(f"unable to write summary: {summary_path}")
    result["output_files"] = [str(p.resolve()) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "rotation_parameters.json"]
    result["output_files"].append(str((output_dir / "rotation_parameters.json").resolve()))
    parameters_path = output_dir / "rotation_parameters.json"
    parameters_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not parameters_path.is_file():
        raise IOError(f"unable to write parameters: {parameters_path}")
    required = ("original.png", "corrected.png", "comparison.png", "fft_diagnostic.png", "edges_before.png", "edges_after.png", "verification_summary.txt", "rotation_parameters.json")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise IOError(f"missing output files: {missing}")
    result["status"] = "ok"
    return result


BATCH_DATASETS = ("20260904-thinkness02_postition1", "T-type")
MASK_NAME = "background_BW.png"
EXCLUDED_IMAGE_NAMES = {"background.png", MASK_NAME}
ROI_SEARCH_MAX_WIDTH = 1024
ROI_SEARCH_MIN_WIDTH = 256
ROI_SEARCH_STEP = 16
MASK_VALID_THRESHOLD = 128


def normalize_mask(mask):
    """Return a boolean validity map: dark first-channel pixels (< 128) are the steel region.

    Boolean input is treated as an already-normalized validity map.
    """
    if mask is None:
        raise ValueError("mask is unreadable")
    if mask.dtype == bool:
        return mask
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D image or a multi-channel image")
    return mask < MASK_VALID_THRESHOLD


def roi_valid_ratio(mask, box):
    """Return the valid-pixel ratio for an xywh ROI, without clipping it."""
    normalized = normalize_mask(mask)
    x, y, width, height = [int(value) for value in box]
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        return 0.0
    if x + width > normalized.shape[1] or y + height > normalized.shape[0]:
        return 0.0
    return float(np.count_nonzero(normalized[y:y + height, x:x + width])) / float(width * height)


def roi_contained(mask, box):
    """Return whether every pixel of an xywh ROI is inside the mask."""
    return roi_valid_ratio(mask, box) == 1.0


def _find_contained_roi(normalized, y_min=0, y_max=None):
    height, width = normalized.shape
    y_max = height if y_max is None else min(height, y_max)
    integral = np.pad(normalized.astype(np.int64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    for roi_width in range(min(ROI_SEARCH_MAX_WIDTH, width), ROI_SEARCH_MIN_WIDTH - 1, -ROI_SEARCH_STEP):
        roi_height = roi_width // 2
        if roi_height > y_max - y_min or roi_width > width:
            continue
        for y in range(y_min, y_max - roi_height + 1):
            row_sums = (integral[y + roi_height, roi_width:] - integral[y, roi_width:]
                        - integral[y + roi_height, :-roi_width] + integral[y, :-roi_width])
            valid_x = np.flatnonzero(row_sums == roi_width * roi_height)
            if valid_x.size:
                x = int(valid_x[0])
                return (x, y, roi_width, roi_height)
    return None


def select_mask_roi(mask):
    """Find three separated contained ROIs when possible, otherwise gate one fallback."""
    normalized = normalize_mask(mask)
    height = normalized.shape[0]
    bands = ((0, height // 3), (height // 3, 2 * height // 3), (2 * height // 3, height))
    boxes = [box for start, end in bands if (box := _find_contained_roi(normalized, start, end)) is not None]
    if not boxes:
        box = _find_contained_roi(normalized)
        if box is None:
            raise ValueError("no mask-contained ROI found from 1024x512 down to 256x128")
        boxes = [box]
    primary = boxes[0]
    return {
        "x": primary[0], "y": primary[1], "width": primary[2], "height": primary[3],
        "xywh": list(primary), "rois_xywh": [list(item) for item in boxes],
        "valid_ratio": roi_valid_ratio(normalized, primary), "roi_count": len(boxes),
        "selection_mode": "three_roi_median" if len(boxes) == 3 else "single_roi_fallback",
        "quality_note": None if len(boxes) == 3 else "Only one fully mask-contained ROI was found; three-ROI policy was not satisfied.",
        "search": {"max_width_px": ROI_SEARCH_MAX_WIDTH, "min_width_px": ROI_SEARCH_MIN_WIDTH,
                   "step_px": ROI_SEARCH_STEP, "aspect_ratio": "2:1",
                   "order": "three vertical bands, descending width, ascending y, ascending x"},
    }


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _batch_image_paths(dataset_dir):
    return sorted(
        (path for path in dataset_dir.glob("*.png") if path.name not in EXCLUDED_IMAGE_NAMES),
        key=lambda path: path.name.casefold(),
    )


def _all_source_signatures(dataset_dir):
    return {
        path.name: {"byte_size": _file_signature(path)[0], "sha256": _file_signature(path)[1]}
        for path in sorted(path for path in dataset_dir.iterdir() if path.is_file())
    }


def _baseline_signatures(data_root, baseline_path, dataset_names):
    if not baseline_path.is_file():
        return {
            "checked": False,
            "valid": False,
            "path": str(baseline_path.resolve()),
            "mismatches": ["baseline_missing"],
            "missing": [],
            "extra": [],
        }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    expected = {
        Path(item["relative_path"].replace("\\", "/")).as_posix(): item
        for item in baseline
        if Path(item["relative_path"].replace("\\", "/")).parts
        and Path(item["relative_path"].replace("\\", "/")).parts[0] in dataset_names
    }
    actual = {}
    for dataset_name in dataset_names:
        dataset_dir = data_root / dataset_name
        for name, signature in _all_source_signatures(dataset_dir).items():
            actual[Path(dataset_name, name).as_posix()] = signature
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = missing + extra
    for key in sorted(set(expected) & set(actual)):
        item = expected[key]
        signature = actual[key]
        if item["byte_size"] != signature["byte_size"] or item["sha256"].lower() != signature["sha256"]:
            mismatches.append(key)
    mismatches = sorted(set(mismatches))
    return {
        "checked": True,
        "valid": not mismatches,
        "path": str(baseline_path.resolve()),
        "mismatches": mismatches,
        "missing": missing,
        "extra": extra,
    }


def _batch_failure(dataset_name, image_path, output_dir, reason, mask_path=None, roi=None, reference=None, calibration_quality=None, measured=None, after_estimate=None, compensation=None, output_files=None, mask_rotation=None, source=None):
    quality = calibration_quality or {
        "status": "quality_failure",
        "residual_definition": "absolute normalized FFT theta in degrees",
        "before_residual_deg": None,
        "after_residual_deg": None,
        "improvement_deg": None,
        "tolerance_deg": QUALITY_TOLERANCE_DEG,
        "normalized_peak_threshold": NORMALIZED_PEAK_THRESHOLD,
        "checks": {
            "finite": False, "peak_magnitude_positive": False, "normalized_peak_above_threshold": False,
            "input_shape": None, "input_shape_valid": False, "input_dtype": None, "input_dtype_valid": False,
            "input_channels": None, "input_channels_valid": False, "strict_residual_improvement": False, "within_tolerance": False,
        },
        "reason": reason,
    }
    result = {
        "status": "failure",
        "dataset": dataset_name,
        "input_path": str(image_path.resolve()) if image_path else None,
        "output_dir": str(output_dir.resolve()),
        "mask_path": str(mask_path.resolve()) if mask_path else None,
        "roi": roi,
        "calibration": reference,
        "calibration_status": quality["status"],
        "calibration_quality": quality,
        "angle_convention": "theta=normalize(atan2(fy, fx)-90); OpenCV positive angle equals the measured theta",
        "fft": {"roi_xywh": roi["xywh"] if roi else None, "measured": measured, "after": after_estimate},
        "measured_angle_deg": measured.get("theta_deg") if measured else None,
        "compensation_angle_deg": compensation,
        "source": source or {"before": None, "after": None, "unchanged": False},
        "mask_rotation": mask_rotation,
        "failure_reasons": [reason],
        "failure_reason": reason,
        "corrected_image_absent": not (output_dir / "corrected.png").exists(),
        "output_files": output_files or [],
    }
    _json_write(output_dir / "result.json", result)
    return result


def _calibration_quality(measured, after_estimate, image=None, measured_rois=None, after_rois=None):
    before_residual = abs(float(measured["theta_deg"]))
    after_residual = abs(float(after_estimate["theta_deg"]))
    measured_rois = measured_rois or [measured]
    after_rois = after_rois or [after_estimate]
    peak_values = [float(item.get("normalized_peak_magnitude", item.get("peak_value", 0.0)))
                   for item in (*measured_rois, *after_rois)]
    finite = all(np.isfinite(float(item[key])) for item in (*measured_rois, *after_rois)
                 for key in ("theta_deg", "peak_value"))
    peak_magnitude_valid = all(np.isfinite(value) and value > NORMALIZED_PEAK_THRESHOLD for value in peak_values)
    shape_valid = image is None or tuple(image.shape) == EXPECTED_IMAGE_SHAPE
    dtype_valid = image is None or image.dtype == np.uint8
    channels_valid = image is None or image.ndim == 2
    improved = after_residual < before_residual
    within_tolerance = after_residual <= QUALITY_TOLERANCE_DEG
    reasons = []
    if not shape_valid: reasons.append(f"input shape must be {EXPECTED_IMAGE_SHAPE}, got {tuple(image.shape)}")
    if not dtype_valid: reasons.append(f"input dtype must be uint8, got {image.dtype}")
    if not channels_valid: reasons.append("input must be single-channel before grayscale conversion")
    if not finite: reasons.append("FFT residual or peak magnitude is not finite")
    if not peak_magnitude_valid: reasons.append(f"normalized FFT peak magnitude must exceed {NORMALIZED_PEAK_THRESHOLD:g}")
    if not improved: reasons.append("candidate compensation did not reduce the absolute FFT residual")
    if not within_tolerance: reasons.append(f"post-correction residual exceeds {QUALITY_TOLERANCE_DEG:.2f}-degree tolerance")
    return {
        "status": "ok" if not reasons else "quality_failure", "residual_definition": "absolute normalized FFT theta in degrees",
        "before_residual_deg": before_residual, "after_residual_deg": after_residual,
        "improvement_deg": before_residual - after_residual, "tolerance_deg": QUALITY_TOLERANCE_DEG,
        "normalized_peak_threshold": NORMALIZED_PEAK_THRESHOLD, "normalized_peak_values": peak_values,
        "checks": {"finite": finite, "peak_magnitude_positive": peak_magnitude_valid,
                   "normalized_peak_above_threshold": peak_magnitude_valid,
                   "input_shape": list(image.shape) if image is not None else None, "input_shape_valid": shape_valid,
                   "input_dtype": str(image.dtype) if image is not None else None, "input_dtype_valid": dtype_valid,
                   "input_channels": _image_channels(image) if image is not None else None, "input_channels_valid": channels_valid,
                   "strict_residual_improvement": improved, "within_tolerance": within_tolerance},
        "reason": "; ".join(reasons) if reasons else None,
    }


def _batch_result(dataset_name, image_path, output_dir, mask_path, roi, reference, measured, compensation, source_before, source_after, corrected, after_estimate, calibration_quality):
    x, y, width, height = roi["xywh"]
    result = {
        "status": "ok",
        "dataset": dataset_name,
        "input_path": str(image_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mask_path": str(mask_path.resolve()),
        "roi": roi,
        "image": {"width": int(corrected.shape[1]), "height": int(corrected.shape[0]), "channels": 1, "dtype": str(corrected.dtype)},
        "calibration": reference,
        "calibration_status": calibration_quality["status"],
        "calibration_quality": calibration_quality,
        "angle_convention": "theta=normalize(atan2(fy, fx)-90); OpenCV positive angle equals the measured theta",
        "fft": {
            "dc_exclusion_radius_px": DC_RADIUS,
            "search_half_plane": "v > 5",
            "window": "2D Hann",
            "roi_xywh": [x, y, width, height],
            "measured": measured,
            "after": after_estimate,
        },
        "measured_angle_deg": measured["theta_deg"],
        "compensation_angle_deg": compensation,
        "rotation": {"interpolation": "INTER_LINEAR", "border": "BORDER_REFLECT_101", "same_size": True},
        "source": {"before": source_before, "after": source_after, "unchanged": source_before == source_after},
        "output_files": [str((output_dir / "corrected.png").resolve()), str((output_dir / "fft_diagnostic.png").resolve())],
    }
    result["verification"] = {
        "output_readable": True,
        "shape_matches_input": True,
        "dtype_matches_input": True,
        "original_unchanged": result["source"]["unchanged"],
    }
    return result


def _write_results_csv(path, results):
    fields = (
        "dataset", "image", "status", "analysis_status", "calibration_status",
        "before_residual_deg", "after_residual_deg", "measured_angle_deg",
        "compensation_angle_deg", "result_json", "corrected_path",
        "analysis_signal_csv", "analysis_signal_png", "failure_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            analysis = result.get("analysis", {})
            writer.writerow({
                "dataset": result.get("dataset"),
                "image": Path(result["input_path"]).name if result.get("input_path") else "",
                "status": result.get("status"),
                "analysis_status": analysis.get("status", "not_run"),
                "calibration_status": result.get("calibration_status", "not_run"),
                "before_residual_deg": result.get("calibration_quality", {}).get("before_residual_deg", ""),
                "after_residual_deg": result.get("calibration_quality", {}).get("after_residual_deg", ""),
                "measured_angle_deg": result.get("measured_angle_deg", ""),
                "compensation_angle_deg": result.get("compensation_angle_deg", ""),
                "result_json": str((Path(result["output_dir"]) / "result.json").resolve()),
                "corrected_path": str((Path(result["output_dir"]) / "corrected.png").resolve()) if result.get("status") == "ok" else "",
                "analysis_signal_csv": analysis.get("signal_csv", ""),
                "analysis_signal_png": analysis.get("signal_png", ""),
                "failure_reason": result.get("failure_reason", ""),
            })


def _write_dataset_outputs(dataset_output, dataset_info, results):
    results_csv = dataset_output / "results.csv"
    results_json = dataset_output / "results.json"
    _write_results_csv(results_csv, results)
    _json_write(results_json, {
        "status": dataset_info["status"],
        "dataset": dataset_info["dataset"],
        "dataset_summary": str((dataset_output / "dataset_summary.json").resolve()),
        "result_count": len(results),
        "results": results,
    })
    dataset_info["results_csv"] = str(results_csv.resolve())
    dataset_info["results_json"] = str(results_json.resolve())
    _json_write(dataset_output / "dataset_summary.json", dataset_info)


def _run_batch_dataset(data_root, output_root, dataset_name):
    dataset_dir = (data_root / dataset_name).resolve()
    dataset_output = (output_root / dataset_name).resolve()
    dataset_output.mkdir(parents=True, exist_ok=True)
    image_paths = _batch_image_paths(dataset_dir)
    mask_path = dataset_dir / MASK_NAME
    source_before = _all_source_signatures(dataset_dir)
    mask = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    analysis_contract = {
        "status": "unavailable",
        "model_status": "not_supplied",
        "reason": "Preprocessing-only scope: no validated N/w/D or KD/MACD/count model is supplied by the project inputs.",
        "method": "masked_horizontal_accumulation",
    }
    dataset_info = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_dir),
        "output_dir": str(dataset_output),
        "mask_path": str(mask_path.resolve()),
        "source_hashes_before": source_before,
        "image_count": len(image_paths),
        "analysis": analysis_contract,
        "assumptions": {
            "input_extension": ".png",
            "excluded_names": sorted(EXCLUDED_IMAGE_NAMES),
            "mask_validity": f"first channel < {MASK_VALID_THRESHOLD} (black = steel region)",
            "roi_search": "descending 2:1 rectangles 1024x512 through 256x128, 16-pixel decrements",
            "rotation": "same-size OpenCV INTER_LINEAR/BORDER_REFLECT_101",
            "horizontal_accumulation": "sum and mean across valid mask pixels for each x coordinate",
        },
    }
    results = []
    roi_diagnostic_path = dataset_output / "mask_roi_diagnostic.png"
    roi_diagnostics_path = dataset_output / "roi_diagnostics.json"
    try:
        roi = select_mask_roi(mask)
    except (ValueError, TypeError) as error:
        reason = str(error)
        if mask is not None:
            save_mask_roi_diagnostic(mask, None, roi_diagnostic_path)
            normalized = normalize_mask(mask)
            roi_diagnostics = {
                "status": "failed",
                "mask_path": str(mask_path.resolve()),
                "mask_shape": list(normalized.shape),
                "valid_pixel_count": int(np.count_nonzero(normalized)),
                "roi": None,
                "roi_contained": False,
                "failure_reason": reason,
            }
        else:
            roi_diagnostics = {"status": "failed", "mask_path": str(mask_path.resolve()), "roi": None, "roi_contained": False, "failure_reason": reason}
        _json_write(roi_diagnostics_path, roi_diagnostics)
        dataset_info.update({
            "status": "failed",
            "failure_reason": reason,
            "roi": None,
            "artifacts": {"mask_roi_diagnostic": str(roi_diagnostic_path.resolve()) if roi_diagnostic_path.is_file() else None, "roi_diagnostics": str(roi_diagnostics_path.resolve())},
        })
        _json_write(dataset_output / "dataset_failure.json", dataset_info)
        for image_path in image_paths:
            source = {"before": {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}, "after": {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}, "unchanged": True}
            results.append(_batch_failure(dataset_name, image_path, dataset_output / image_path.stem, reason, mask_path, source=source))
        source_after = _all_source_signatures(dataset_dir)
        dataset_info["source_hashes_after"] = source_after
        dataset_info["source_unchanged"] = source_before == source_after
        dataset_info["success_count"] = 0
        dataset_info["failure_count"] = len(results)
        _write_dataset_outputs(dataset_output, dataset_info, results)
        return dataset_info, results

    save_mask_roi_diagnostic(mask, roi, roi_diagnostic_path)
    normalized_mask = normalize_mask(mask)
    _json_write(roi_diagnostics_path, {
        "status": "ok",
        "mask_path": str(mask_path.resolve()),
        "mask_shape": list(normalized_mask.shape),
        "valid_pixel_count": int(np.count_nonzero(normalized_mask)),
        "roi": roi,
        "roi_count": roi.get("roi_count", 1),
        "selection_mode": roi.get("selection_mode", "single_roi_fallback"),
        "quality_note": roi.get("quality_note"),
        "roi_contained": roi_contained(normalized_mask, roi["xywh"]),
    })
    dataset_info["roi"] = roi
    dataset_info["artifacts"] = {
        "mask_roi_diagnostic": str(roi_diagnostic_path.resolve()),
        "roi_diagnostics": str(roi_diagnostics_path.resolve()),
    }
    reference = None
    reference_path = dataset_dir / "096.png" if dataset_name == BATCH_DATASETS[0] else None
    if reference_path:
        reference_image = read_image(reference_path, cv2.IMREAD_GRAYSCALE)
        if reference_image is None or tuple(reference_image.shape) != tuple(mask.shape[:2]):
            reference = {"mode": "reference_096", "path": str(reference_path.resolve()), "status": "failed"}
        else:
            boxes = roi.get("rois_xywh", [roi["xywh"]])
            reference_estimates, reference_estimate = estimate_boxes(reference_image, boxes)
            reference = {"mode": "reference_096", "path": str(reference_path.resolve()), "status": "ok", "angle_deg": reference_estimate["theta_deg"], "fft": {"rois": reference_estimates, "median": reference_estimate}}
            reference_candidate = rotate_image(reference_image, reference_estimate["theta_deg"])
            reference_diagnostic = dataset_output / "reference_096_diagnostic.png"
            save_comparison(reference_image, reference_candidate, reference_diagnostic, reference_estimate["theta_deg"])
            reference["diagnostic_png"] = str(reference_diagnostic.resolve())
            reference_after_estimates, reference_after_estimate = estimate_boxes(reference_candidate, boxes)
            reference["quality"] = _calibration_quality(reference_estimate, reference_after_estimate, reference_image, reference_estimates, reference_after_estimates)
            if reference.get("quality", {}).get("status") != "ok":
                reference["status"] = "quality_failure"
                reference["usable"] = False
            reference["angle_convention"] = "theta=normalize(atan2(fy, fx)-90); OpenCV positive angle equals the measured theta"
        reference["calibration_status"] = reference.get("quality", {}).get("status", reference["status"])
        reference["source"] = {"byte_size": _file_signature(reference_path)[0], "sha256": _file_signature(reference_path)[1]}
        reference["roi_xywh"] = roi["xywh"]
        reference["result_path"] = str((dataset_output / "reference_096.json").resolve())
        _json_write(dataset_output / "reference_096.json", {
            "dataset": dataset_name,
            "reference_image": reference,
            "traceability": "096.png calibration is reused for every position1 image",
        })
        dataset_info["reference_output"] = reference["result_path"]
    elif dataset_name == BATCH_DATASETS[1]:
        reference = {"mode": "per_image", "status": "ok"}

    dataset_info["reference_calibration"] = reference
    for image_path in image_paths:
        image_output = dataset_output / image_path.stem
        image_output.mkdir(parents=True, exist_ok=True)
        source_signature_before = {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}
        measured = None
        after_estimate = None
        compensation = None
        reference_gate_failure = False
        try:
            image_raw = read_image(image_path, cv2.IMREAD_UNCHANGED)
            if image_raw is None:
                raise ValueError("unable to read image")
            if image_raw.ndim != 2 or image_raw.dtype != np.uint8:
                raise ValueError(f"unsupported input channels/dtype: channels={_image_channels(image_raw)}, dtype={image_raw.dtype}; expected 1/uint8")
            gray = image_raw
            if tuple(gray.shape) != tuple(mask.shape[:2]):
                raise ValueError(f"image shape {tuple(gray.shape)} does not match mask shape {tuple(mask.shape[:2])}")
            boxes = roi.get("rois_xywh", [roi["xywh"]])
            measured_rois, measured = estimate_boxes(gray, boxes)
            if reference and reference.get("mode") == "reference_096":
                if reference.get("status") != "ok" or reference.get("calibration_status") != "ok":
                    reference_gate_failure = True
                    compensation = float(reference.get("angle_deg", measured["theta_deg"]))
                else:
                    compensation = float(reference["angle_deg"])
            else:
                compensation = float(measured["theta_deg"])
            corrected = rotate_image(gray, compensation)
            after_rois, after_estimate = estimate_boxes(corrected, boxes)
            calibration_quality = _calibration_quality(measured, after_estimate, gray, measured_rois, after_rois)
            if reference_gate_failure:
                calibration_quality["status"] = "quality_failure"
                reference_reason = "096.png reference calibration failed quality gate"
                calibration_quality["reason"] = "; ".join(filter(None, (reference_reason, calibration_quality["reason"])))
            save_fft_diagnostic(gray, [tuple(box) for box in boxes], measured_rois, image_output / "fft_diagnostic.png")
            rotated_mask_path = image_output / "background_BW_rotated.png"
            write_image(rotated_mask_path, rotate_mask(mask, compensation))
            mask_rotation = {
                "input_path": str(mask_path.resolve()),
                "output_path": str(rotated_mask_path.resolve()),
                "angle_deg": compensation,
                "interpolation": "INTER_NEAREST",
                "border": "BORDER_CONSTANT",
                "same_size": True,
            }
            corrected_path = image_output / "corrected.png"
            if calibration_quality["status"] != "ok":
                for stale_name in (
                    "corrected.png",
                    *[f"masked_horizontal_accumulation_roi{index}.{extension}"
                      for index in range(1, roi.get("roi_count", 1) + 1)
                      for extension in ("csv", "png")],
                ):
                    stale_path = image_output / stale_name
                    if stale_path.is_file():
                        stale_path.unlink()
                source_signature_after = {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}
                result = _batch_failure(
                    dataset_name,
                    image_path,
                    image_output,
                    "calibration quality failure: " + calibration_quality["reason"],
                    mask_path,
                    roi,
                    reference,
                    calibration_quality=calibration_quality,
                    measured=measured,
                    after_estimate=after_estimate,
                    compensation=compensation,
                    output_files=[str((image_output / "fft_diagnostic.png").resolve()), str(rotated_mask_path.resolve())],
                    mask_rotation=mask_rotation,
                    source={"before": source_signature_before, "after": source_signature_after, "unchanged": source_signature_before == source_signature_after},
                )
                result["fft"]["rois"] = measured_rois
                result["fft"]["after_rois"] = after_rois
                result["analysis"] = {
                    "status": "unavailable",
                    "model_status": "not_supplied",
                    "execution_status": "not_run",
                    "reason": calibration_quality["reason"],
                }
                result["analysis_status"] = "unavailable"
                _json_write(image_output / "result.json", result)
                results.append(result)
                continue
            write_image(corrected_path, corrected)
            readable = read_image(corrected_path, cv2.IMREAD_UNCHANGED)
            if readable is None or readable.shape != gray.shape or readable.dtype != gray.dtype:
                raise IOError("corrected image shape or dtype does not match input")
            rotated_mask = rotate_mask(mask, compensation)
            effective_roi = select_mask_roi(rotated_mask)
            if not all(roi_contained(rotated_mask, box) for box in effective_roi.get("rois_xywh", [effective_roi["xywh"]])):
                raise ValueError("post-rotation ROI is not fully contained in rotated mask")
            analysis = _write_analysis_artifacts(readable, rotated_mask, image_output, effective_roi)
            source_signature_after = {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}
            result = _batch_result(dataset_name, image_path, image_output, mask_path, roi, reference, measured, compensation, source_signature_before, source_signature_after, corrected, after_estimate, calibration_quality)
            result["effective_roi"] = effective_roi
            result["mask_rotation"] = mask_rotation
            result["fft"]["rois"] = measured_rois
            result["fft"]["after_rois"] = after_rois
            result["analysis"] = analysis
            result["analysis_status"] = analysis["status"]
            result["output_files"].extend([analysis["signal_csv"], analysis["signal_png"], str(rotated_mask_path.resolve()), str((image_output / "result.json").resolve())])
            _json_write(image_output / "result.json", result)
        except (IOError, OSError, ValueError, cv2.error) as error:
            source_signature_after = {"byte_size": _file_signature(image_path)[0], "sha256": _file_signature(image_path)[1]}
            existing_outputs = [
                str(path.resolve()) for path in (image_output / "fft_diagnostic.png", image_output / "background_BW_rotated.png")
                if path.is_file()
            ]
            result = _batch_failure(
                dataset_name,
                image_path,
                image_output,
                str(error),
                mask_path,
                roi,
                reference,
                calibration_quality=reference.get("quality") if reference and reference.get("mode") == "reference_096" else None,
                measured=measured if "measured" in locals() else None,
                after_estimate=after_estimate if "after_estimate" in locals() else None,
                compensation=compensation if "compensation" in locals() else None,
                output_files=existing_outputs,
                source={"before": source_signature_before, "after": source_signature_after, "unchanged": source_signature_before == source_signature_after},
            )
            result["analysis"] = {
                "status": "unavailable",
                "model_status": "not_supplied",
                "execution_status": "not_run",
                "reason": str(error),
            }
            result["analysis_status"] = "unavailable"
            _json_write(image_output / "result.json", result)
        results.append(result)

    source_after = _all_source_signatures(dataset_dir)
    dataset_info.update({
        "status": "ok" if all(item["status"] == "ok" for item in results) and source_before == source_after else "failed",
        "source_hashes_after": source_after,
        "source_unchanged": source_before == source_after,
        "success_count": sum(item["status"] == "ok" for item in results),
        "failure_count": sum(item["status"] != "ok" for item in results),
    })
    _write_dataset_outputs(dataset_output, dataset_info, results)
    return dataset_info, results


def run_batch(data_root, output_root):
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_signatures(data_root, output_root / "data_baseline_before.json", BATCH_DATASETS)
    summaries = []
    all_results = []
    for dataset_name in BATCH_DATASETS:
        summary, results = _run_batch_dataset(data_root, output_root, dataset_name)
        summaries.append(summary)
        all_results.extend(results)
    _write_results_csv(output_root / "results.csv", all_results)
    result = {
        "status": "ok" if all(summary["status"] == "ok" for summary in summaries) and baseline["mismatches"] == [] else "failed",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "datasets": list(BATCH_DATASETS),
        "baseline": baseline,
        "dataset_summaries": [summary for summary in summaries],
        "results_csv": str((output_root / "results.csv").resolve()),
        "result_count": len(all_results),
    }
    _json_write(output_root / "batch_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.batch:
        if args.output_root is None:
            parser.error("--output-root is required with --batch")
        print(json.dumps(run_batch(args.data_root, args.output_root), ensure_ascii=False, indent=2))
        return
    if args.input is None:
        parser.error("--input is required unless --batch is used")
    if args.parameters or args.output or args.metadata:
        if not all((args.parameters, args.output, args.metadata)):
            parser.error("--parameters, --output, and --metadata must be provided together")
        result = run_with_parameters(args.input, args.parameters, args.output, args.metadata)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required when applying estimated correction")
    result = run(args.input, args.output_dir)
    print(json.dumps({k: result[k] for k in ("estimated_angle_deg", "compensation_angle_deg", "before_abs_median_angle_deg", "after_abs_median_angle_deg")}, indent=2))


if __name__ == "__main__":
    main()
