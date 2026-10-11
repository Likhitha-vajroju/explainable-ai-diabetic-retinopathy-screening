# Preprocessing

`dataset.py` reads the Kaggle APTOS 2015/2019 preprocessed CSV labels, resolves image paths, loads individual images, and generates an audit summary. `splitting.py` creates reproducible, stratified training/validation splits, groups IDs with a recognized laterality suffix, and preserves the published test split. IDs without that suffix use sample-level groups and are counted in the summary because patient-level grouping cannot be inferred for them. `visualization.py` displays sample images with their stored labels.

The pipeline keeps the CSV's `diagnosis` values unchanged by default. Any label mapping must be supplied explicitly and must cover each observed label. See [`data/README.md`](../../data/README.md) for dataset placement, provenance caveats, and usage.
