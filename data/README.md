# Dataset setup and inspection

## Source and inspected format

The selected source is [APTOS 2015&2019 Preprocessed on Kaggle](https://www.kaggle.com/datasets/alinusik/aptos-20152019-preprocessed). Its published file listing contains:

- `trainData.csv` (`id_code`, `diagnosis`) and 7,243 PNG files under `trainImgs/trainImgs/`.
- `testData.csv` (`id_code`, `diagnosis`) and 2,500 PNG files under `testImgs/testImgs/`.
- 9,743 labeled rows and 9,743 PNG files in total.

The observed raw `diagnosis` counts are:

| Source | `0` | `1` | `2` | `3` | `4` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,805 | 1,670 | 1,699 | 1,066 | 1,003 |
| Test | 500 | 500 | 500 | 500 | 500 |
| Total | 2,305 | 2,170 | 2,199 | 1,566 | 1,503 |

These are **raw CSV values**, not verified clinical class definitions. The Kaggle description does not define the codes, and its metadata reports the license as **Unknown**. The pipeline therefore does not map them to DR severity levels. Confirm the label definitions and permitted use with authoritative source documentation before interpreting or distributing results.

The identifier format is mixed. All 2,500 test IDs and 3,581 train IDs use a `<prefix>_left` / `<prefix>_right` suffix; the other 3,662 train IDs are 12-character opaque IDs. The default configuration groups matching laterality IDs by their shared prefix. Opaque IDs fall back to sample-level groups because no patient identifier is exposed for them; therefore, patient-level leakage cannot be ruled out for those samples. The shared-prefix grouping is inferred from the identifier format, not independently confirmed as a patient identifier, and should be checked against source documentation. The published test source is preserved as test data. A local audit is still needed to establish actual image dimensions, readability, corruption, and byte-identical duplicates.

## Place the dataset

Download and extract the dataset yourself after checking its terms. Keep the original files together in this layout:

```text
data/
  aptos-20152019-preprocessed/
    trainData.csv
    trainImgs/
      trainImgs/
        *.png
    testData.csv
    testImgs/
      testImgs/
        *.png
```

The dataset contents are excluded from Git. The pipeline reads files in place and writes only summary/split metadata; it does not copy, modify, or delete source images. You can store the dataset elsewhere and pass that path to `--dataset-root`.

## Run the audit and create split metadata

Install the image and plotting dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Run from the repository root, changing paths as needed:

```powershell
python -m src.preprocessing.dataset_cli `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --output-dir "data/metadata" `
  --seed 42 `
  --validation-fraction 0.15
```

The command writes:

- `dataset_summary.json`: overall and source-specific label counts/imbalance, actual image formats and dimensions, missing/corrupted/unreadable files, duplicate identifiers/content, and unlabeled image files.
- `split_metadata.csv`: each sample's relative image path, original label, group ID, source, and assigned split.
- `split_config.json`: source paths, configured columns, grouping expression, seed, and split parameters. `dataset_summary.json` reports the number of sample-level grouping fallbacks.

The summary fully decodes each image and computes SHA-256 hashes to detect duplicate content, so running it reads every image. If any image is missing, corrupted, or unreadable, the summary is retained and split files are not produced. Split ratios can vary slightly because groups (and images within them) cannot be divided. The test source is retained unchanged; only the published training source is divided into train and validation.

To display labeled examples in a Python session:

```python
from src.preprocessing.dataset import aptos_preprocessed_config, load_dataset
from src.preprocessing.visualization import show_samples

records = load_dataset(aptos_preprocessed_config("data/aptos-20152019-preprocessed"))
show_samples(records, count=8, seed=42)
```

Run the pipeline tests from the repository root with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
