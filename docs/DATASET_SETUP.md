# Dataset setup

## Phish-IRIS
Download `phishIRIS_DL_Dataset` from Mendeley and extract below `data/raw/phish_iris`. The adapter recursively detects image files and uses the immediate parent folder as the brand. The `other` class is recorded as legitimate.

## Putra Zenodo dataset
For a first experiment download `phishing.csv` and one `phishing_XXXX-XXXX.zip` archive. Extract screenshots below `data/raw/putra`. The adapter recursively finds images and marks paths containing `not-phishing` as legitimate. Because release layouts can differ, inspect `manifest.csv` after preparation.

## Labels
These sources do not contain the four paper intention ground-truth labels. Add them through the annotation UI. Do not convert brand or phishing/legitimate status into intention ground truth.
