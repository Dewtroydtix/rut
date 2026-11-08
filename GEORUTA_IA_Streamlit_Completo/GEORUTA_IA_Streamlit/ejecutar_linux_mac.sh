#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null; then
  echo "Se necesita Python 3.10 o superior."
  exit 1
fi
if [ ! -f ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
