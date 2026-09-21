# Quick Start

```bash
# 1 · dependencies
pip install -r requirements.txt

# 2 · Ollama (keep running in its own terminal)
ollama serve

# 3 · models - all multimodal; none from the paper; none LLaMA
ollama pull minicpm-v:8b
ollama pull granite3.2-vision:2b
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b

# 4 · datasets  ->  see data/raw/README.md
#     data/raw/putra/phishing/<record-id>/screenshots/*.png
#     data/raw/phishIris/train/<brand>/*.png

# 5 · verify, then launch
python scripts/check_setup.py
./run_ui.sh            # Windows: run_ui.bat
```

Open http://localhost:8501

---

## The workflow (order matters)

| # | Step | Page / command |
|---|---|---|
| 1 | Confirm both datasets load | **📚 Dataset Explorer** |
| 2 | **Annotators build the manifest** | **📋 Manifest Builder** → *Build manifest* |
| 3 | Review anything flagged ⚠️ | **✍️ Manual Annotation** → queue *Manifest flagged for review* |
| 4 | **Framework predicts** on the same samples | **🤖 Framework Predictions** → pool *manifest-covered only* |
| 5 | Run the single-agent baseline too | same page, pipeline = *single-agent* |
| 6 | **Score predictions vs the manifest** | **📊 Evaluation** |

Command-line equivalent:

```bash
python scripts/build_manifest.py   --n 50
python scripts/run_predictions.py  --n 50 --scope manifest
python scripts/run_predictions.py  --n 50 --scope manifest --mode single
python scripts/run_evaluation.py   --errors
```

---

## Low-VRAM profile

Edit `config.yaml` — keep every model multimodal:

```yaml
models:
  annotator_a:         {model: "moondream",            vision: true, temperature: 0.1}
  annotator_b:         {model: "granite3.2-vision:2b", vision: true, temperature: 0.1}
  manifest_tiebreaker: {model: "gemma3:4b",            vision: true, temperature: 0.0}
  manifest_finalizer:  {model: "gemma3:4b",            vision: true, temperature: 0.0}
  vision:              {model: "moondream",            vision: true, temperature: 0.1}
  context:             {model: "gemma3:4b",            vision: true, temperature: 0.0}
  classifier:          {model: "gemma3:4b",            vision: true, temperature: 0.0}
  specialist:          {model: "gemma3:4b",            vision: true, temperature: 0.0}
  validator:           {model: "gemma3:4b",            vision: true, temperature: 0.0}
ollama:
  num_ctx: 4096
framework:
  max_image_edge: 896
```

## No datasets yet?

Use the **Upload** tab on the Analyse page — any website screenshot works.

## Verify without Ollama

```bash
python tests/test_pipeline.py
python tests/test_end_to_end_mock.py
```

The second test prints a VLM audit proving every agent call carried the image.
