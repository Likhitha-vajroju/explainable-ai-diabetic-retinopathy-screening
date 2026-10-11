# Image quality assessment

This module is a configurable **engineering screening gate** for retinal fundus
images. It provides measurements and reasons that can help a caller decide
whether an image should be reviewed or recaptured instead of being passed
blindly to a later model. This repository does not yet implement a DR
classifier, Grad-CAM, or a frontend.

## Measurements

`assess_image()` returns five measurements, each with its raw value, status,
normalized contribution to the 0-100 score, and explanatory detail:

| Measurement | Calculation |
| --- | --- |
| Focus | Variance of a Pillow edge-filter response inside the detected retinal field. |
| Illumination | Mean grayscale intensity inside the detected retinal field. |
| Contrast | Grayscale standard deviation inside the detected retinal field. |
| Field of view | Area of the detected non-background bounding box and its width/height relative to the image. |
| Artifacts | Fraction of detected field pixels near pure black or white, as a basic clipping indicator. If no field is detected, the full frame is used. This does not detect every artifact type. |

Each indicator is `PASS`, `BORDERLINE`, or `FAIL`. The overall score is the
rounded mean of the five indicator scores. Any failed indicator, or a score
below the configured ungradable score, makes the result `UNGRADABLE`.
Otherwise, a result is `ACCEPTABLE` only when every indicator passes and the
score meets the configured acceptable score; remaining results are
`BORDERLINE`. Poor-quality indicators are included in `reasons`.

**No cutoff or score here is medically validated.** Defaults are illustrative
engineering heuristics only; they must be tuned and validated on
representative cameras, populations, acquisition workflows, and expert
gradability labels before any clinical use. This output is not a diagnosis or
a substitute for human review.

## Run on one image

From the repository root:

```powershell
python -m src.image_quality_assessment.cli "path\to\fundus.jpg"
```

For a machine-readable report and an annotated visual companion:

```powershell
python -m src.image_quality_assessment.cli "path\to\fundus.jpg" --json --save-visualization "quality-review.png"
```

The saved PNG retains the input image and adds a status-colored score and
measurement summary banner. It is a visual aid only, not a segmentation or
clinical interpretation.

## Configure the heuristic cutoffs

The API accepts a `QualityThresholds` instance. The CLI accepts a JSON object
whose keys override any subset of the dataclass fields:

```powershell
python -m src.image_quality_assessment.cli "path\to\fundus.jpg" --thresholds-json "quality-thresholds.json"
```

Example `quality-thresholds.json`:

```json
{
  "focus_min_variance": 24.0,
  "brightness_min_mean": 50.0,
  "brightness_max_mean": 205.0,
  "contrast_min_stddev": 28.0,
  "fov_min_area_fraction": 0.22,
  "clipped_max_fraction": 0.08
}
```

All unspecified fields retain their defaults. The thresholds are checked for
valid ordering and range consistency. They should be selected using a
documented validation protocol, not interpreted as universal or clinically
established cutoffs.

## Python API

```python
from src.image_quality_assessment import QualityThresholds, assess_image

result = assess_image(
    "path/to/fundus.jpg",
    QualityThresholds(focus_min_variance=24.0),
)
print(result.format_report())
print(result.to_dict())
```

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -p "test_image_quality_assessment.py" -v
```
