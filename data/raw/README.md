# Where to put the datasets

Both sources are expected here, each keeping its own original structure. The
loaders normalise them into one `Sample` schema, so nothing has to be renamed.

## 1. Putra phishing website dataset

    data/raw/putra/
      phishing/
        <record-id>/
          screenshots/   *.png
          assets/        (optional: url.txt, meta.json, extra images)
      not-phishing/
        <record-id>/
          screenshots/   *.png
          assets/

* `<record-id>` is the original ObjectId-style folder name.
* Only `phishing/` is used for intention analysis by default
  (`datasets.putra.phishing_only: true` in `config.yaml`).
* If a record has several screenshots, the loader picks the most
  representative one and keeps the rest as metadata.
* A `url.txt` or `meta.json` inside `assets/` is harvested automatically.

## 2. Phish-IRIS dataset

    data/raw/phishIris/
      train/
        apple/ amazon/ chase/ dhl/ facebook/ ... other/     *.png
      val/
        apple/ amazon/ chase/ dhl/ facebook/ ... other/     *.png

* Every image is a phishing screenshot.
* The folder name is the impersonated brand and is used as a weak sector hint
  (e.g. `chase` -> financial, `dhl` -> delivery & couriers).

## Verifying the layout

    python scripts/check_setup.py

Both paths can be changed in `config.yaml` under `datasets:` if you keep the
data elsewhere.
