# Stage 1: Builder
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir --default-timeout=1000 -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy Code
COPY . .

# Permissions
RUN chmod +x scripts/start_server.sh

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["./scripts/start_server.sh"]