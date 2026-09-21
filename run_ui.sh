#!/usr/bin/env bash
# Launch the PhishIntentionLLM console.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "WARNING: Ollama does not appear to be running on http://localhost:11434"
  echo "         Start it in another terminal with:  ollama serve"
  echo
fi

exec streamlit run app/streamlit_app.py "$@"
