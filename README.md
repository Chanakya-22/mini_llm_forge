# Mini-LLM-Forge 

A containerized, modular MLOps pipeline designed for the efficient fine-tuning (QLoRA) and deployment of lightweight Large Language Models (1B-3B parameters) on consumer-grade hardware.

This project bridges the gap between experimental Jupyter notebooks and production-grade inference systems by separating the ML engine, the API gateway, and the frontend UI into distinct, manageable layers.

---

##  System Architecture

The forge operates on a decoupled architecture, allowing hot-swapping of models and frontend interfaces without rebuilding the core engine.

1. **The Engine (Backend):** Utilizes `transformers`, `peft`, and `trl` to load base models into 4-bit quantized memory, apply LoRA adapters, and execute Supervised Fine-Tuning (SFT).
2. **The API Gateway:** A high-performance FastAPI server running on Uvicorn/Gunicorn workers, utilizing Pydantic for strict data contract validation.
3. **The Frontend:** A lightweight, decoupled Streamlit chat interface that communicates with the backend via REST.
4. **The Operations Layer:** A multi-stage Docker environment ensuring CUDA dependencies are perfectly isolated from the host machine.

---

##  Tech Stack & Dependencies

**Core Engine:**
* PyTorch (Hardware-accelerated via `bfloat16`)
* Hugging Face (`transformers`, `datasets`, `accelerate`)
* `peft` & `bitsandbytes` (QLoRA 4-bit Quantization)
* `trl` (SFTTrainer & SFTConfig)

**Serving & UI:**
* FastAPI & Pydantic
* Uvicorn & Gunicorn
* Streamlit & Requests

**Infrastructure:**
* Docker & Docker Compose
* NVIDIA CUDA Toolkit

---

##  Project Structure

```text
mini-llm-forge/
├── config/              
│   ├── .env             # (Git-ignored) Environment configurations
│   └── logging.yaml     # System logging configurations
├── data/                
│   └── custom_data.jsonl # Training dataset (JSON Lines format)
├── models/              # (Git-ignored) Local storage for base models
├── model_output/        # (Git-ignored) Trained LoRA adapters
├── scripts/             
│   └── start_server.sh  # Boot script for Gunicorn/Uvicorn
├── src/                 
│   ├── app/             # FastAPI routes and schemas
│   ├── core/            # Configuration loaders
│   ├── engine/          # Model loading and inference logic
│   ├── frontend/        
│   │   └── ui.py        # Streamlit chat interface
│   └── train.py         # QLoRA fine-tuning execution script
├── Dockerfile           # Runtime environment definition
└── requirements.txt     # Python dependencies


Follow these steps to build, train, and serve your own custom LLM from scratch.

1. Prerequisites
Docker Desktop installed and running.

An NVIDIA GPU (8GB+ VRAM recommended) with updated drivers.

Python 3.10+ installed locally for the frontend.

2. Initial Setup
Clone the repository and set up your environment variables.

Bash
git clone [https://github.com/YourUsername/mini_llm_forge.git](https://github.com/YourUsername/mini_llm_forge.git)
cd mini_llm_forge
Create a .env file inside the config/ directory:

Ini, TOML
# config/.env
PROJECT_NAME=Mini-LLM-Forge
API_V1_STR=/api/v1
BASE_MODEL_NAME=/app/models/tinyllama
ADAPTER_PATH=model_output
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
3. Download the Base Model (Offline Mode)
To prevent container bloat and bypass network restrictions inside Docker, download the base model locally first.

Bash
mkdir models
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; AutoTokenizer.from_pretrained(model_id).save_pretrained('models/tinyllama'); AutoModelForCausalLM.from_pretrained(model_id).save_pretrained('models/tinyllama')"
4. Build the Engine
Construct the isolated Linux environment containing all CUDA and PyTorch dependencies.

Bash
docker build -t mini-llm-forge:v1 .
5. Train the Model (QLoRA)
Add your formatted facts to data/custom_data.jsonl. Then, execute the fine-tuning script. This command utilizes volume mounts to allow live code editing and saves the trained adapter directly to your host machine.

Bash
docker run --gpus all \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/model_output:/app/model_output \
  -v ${PWD}/src:/app/src \
  --env-file config/.env \
  mini-llm-forge:v1 python src/train.py
6. Serve the API
Once training is complete, boot up the FastAPI inference server. The engine will automatically detect and merge your new LoRA adapter with the base model on the fly.

Bash
docker run --gpus all \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/model_output:/app/model_output \
  -v ${PWD}/src:/app/src \
  -p 8000:8000 \
  --env-file config/.env \
  --name llm-server \
  mini-llm-forge:v1
API Documentation available at: http://localhost:8000/docs

7. Launch the UI
Open a new local terminal (outside of Docker), install the frontend dependencies, and launch Streamlit.

Bash
pip install streamlit requests
python -m streamlit run src/frontend/ui.py
Access the chat interface at: http://localhost:8501

 Current Limitations & Roadmap
Data Scarcity: The model currently relies on a micro-dataset. To prevent base-model hallucinations, the custom_data.jsonl file must be expanded significantly.

Context Window: Currently hard-gated at 512 tokens during training for memory efficiency.

Formatting: Requires strict adherence to <human>: and <bot>: tagging logic during inference to prevent infinite generation loops.
