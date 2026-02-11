#!/bin/bash
# Make sure we are in the root directory
cd "$(dirname "$0")/.."

# Set Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run Gunicorn with Uvicorn Workers
# Workers = 1 (Because loading multiple LLMs on one GPU crashes it)
exec gunicorn src.app.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300