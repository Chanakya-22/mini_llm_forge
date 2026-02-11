    # Mini-LLM-Forge

Mini-LLM-Forge is a modular MLOps pipeline designed for the efficient fine-tuning and deployment of lightweight Large Language Models (LLMs). It provides a standardized architecture for training models using Parameter-Efficient Fine-Tuning (PEFT) and serving them via a high-performance, containerized API.

The primary objective of this project is to bridge the gap between experimental scripts and production-grade inference systems, enabling the deployment of custom-trained 1B-3B parameter models on consumer-grade hardware or edge devices.

## System Architecture

The project is structured into three distinct layers to ensure scalability and separation of concerns:

1.  **The Engine Layer (`src/engine`):** Manages the lifecycle of the LLM, including 4-bit quantization (QLoRA), adapter merging, and memory management. It isolates GPU operations from the web server logic.
2.  **The API Layer (`src/app`):** A FastAPI-based gateway that handles request validation, schema definition, and routing. It adheres to strict data contracts using Pydantic.
3.  **The Operations Layer (`Docker` & `scripts`):** Provides a reproducible runtime environment using multi-stage Docker builds and Gunicorn process management.

## Project Structure

```text
mini-llm-forge/
├── config/              # Configuration and environment variables
├── data/                # Training datasets (JSONL format)
├── scripts/             # Operational scripts for training and serving
├── src/                 # Source code
│   ├── app/             # API routes and Pydantic schemas
│   ├── core/            # Logging and configuration loaders
│   └── engine/          # Model loading and inference logic
├── Dockerfile           # Multi-stage container definition
└── requirements.txt     # Python dependencies