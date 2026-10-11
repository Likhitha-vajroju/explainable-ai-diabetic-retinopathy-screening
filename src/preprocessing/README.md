# Retinal image preprocessing

`image_pipeline.py` provides reusable, configurable preprocessing for a single
fundus photograph. It returns three separate representations: a detached,
EXIF-oriented RGB copy of the input, an 8-bit RGB enhanced image, and (by default)
a NumPy RGB `float32` array scaled to `[0, 1]`. Processing never writes to or
changes the input file.

## Processing steps and rationale

1. **Image loading and resizing:** EXIF orientation is applied and pixels are
   detached from the source file. The default 512 x 512 output uses
   aspect-preserving fit with black padding instead of stretching the retina.
   Configure `output_size=None` to keep the source dimensions, or disable
   `preserve_aspect_ratio` when an exact-size stretch is explicitly wanted.
2. **Retinal region/background handling:** a configurable grayscale threshold
   separates non-dark pixels from dark background. `background_mode="black"`
   masks the background in the enhanced image; `"preserve"` leaves its input
   pixels alone. This simple threshold is not a retinal segmentation model;
   unusually dark fields or non-black backgrounds may need a different
   threshold or a preserved background.
3. **Illumination normalization:** a low-frequency Gaussian estimate is used
   to correct slow brightness variation within the detected field. This is
   intended to reduce uneven lighting while limiting changes to the
   luminance channel. Set `normalize_illumination=False` to skip it.
4. **CLAHE/local contrast:** a contrast-limited adaptive histogram equalizer
   operates on luminance in a configurable tile grid. Limiting each tile's
   histogram reduces excessive local amplification compared with unconstrained
   equalization. Set `apply_clahe=False` to skip it.
5. **Noise reduction:** an optional mild Gaussian filter is applied to
   luminance to reduce high-frequency noise after contrast enhancement. Its
   radius is configurable; `0` disables the filter. Strong smoothing can remove
   fine detail, so use conservative settings and inspect results.
6. **Pixel normalization:** the enhanced RGB image is converted to float32
   `[0, 1]` by default, ready for downstream numerical pipelines. Set
   `normalize_pixels=False` to receive float32 `[0, 255]` values instead.

Enhancement is experimental preprocessing, **not evidence of improved
diagnostic accuracy**. Resizing, masking, normalization, local contrast
enhancement, and denoising can alter image appearance or remove details.
Compare outputs visually and validate a chosen configuration against the
intended acquisition data and task before using it in later research. This
module does not train or run a classifier and does not provide clinical
interpretation.

## Run the one-image demo

From the repository root in PowerShell:

```powershell
python -m src.preprocessing.image_demo "C:\path\to\fundus.jpg" --output-dir "C:\path\to\review-output"
```

The output directory receives three distinct files:

- `original.<input extension>`: byte-for-byte copy of the source.
- `enhanced.png`: processed image.
- `comparison.png`: labeled side-by-side visualization.

The demo refuses to overwrite any of these output paths. If `--output-dir` is
omitted, a new `preprocessed_<image stem>` directory is created beside the
input. Default enhanced dimensions are 512 x 512. Useful options:

```powershell
python -m src.preprocessing.image_demo "C:\path\to\fundus.jpg" --width 768 --height 768 --background-mode preserve
python -m src.preprocessing.image_demo "C:\path\to\fundus.jpg" --disable-clahe --disable-illumination-normalization
```

## Python API

```python
from src.preprocessing import PreprocessingConfig, preprocess_image

result = preprocess_image(
    "path/to/fundus.jpg",
    PreprocessingConfig(
        output_size=(512, 512),
        background_mode="black",
        clahe_clip_limit=2.0,
        noise_reduction_radius=0.6,
    ),
)
result.original_image.save("original-copy.png")
result.enhanced_image.save("enhanced.png")
model_input = result.normalized_image  # RGB float32, [0, 1] by default
```

Run the focused tests from the repository root:

```powershell
python -m unittest discover -s tests -p "test_image_preprocessing.py" -v
```

## Other preprocessing utilities

`dataset.py` reads the configured APTOS 2015/2019 preprocessed CSV labels,
resolves image paths, loads individual images, and generates an audit summary.
`splitting.py` creates reproducible, stratified train/validation splits,
groups IDs with a recognized laterality suffix, and preserves the published
test split. `visualization.py` displays dataset samples with their stored
labels. The CSV's `diagnosis` values are kept unchanged by default; label
mapping must be explicit. See [`data/README.md`](../../data/README.md) for
dataset placement and provenance caveats.
