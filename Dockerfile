FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/models/llama-3.2-1b-instruct.Q4_K_M.gguf" \
    PORT=8000

WORKDIR /app

# Instalar herramientas del sistema requeridas para la compilación de C/C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descargar modelo GGUF en la capa de construcción
RUN mkdir -p /app/models && \
    curl -L -o /app/models/llama-3.2-1b-instruct.Q4_K_M.gguf \
    https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Pre-descargar modelo de embeddings en la caché interna
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copiar archivos fuente y datos
COPY data/ ./data/
COPY src/ ./src/

EXPOSE 8000

# Verificación de salud interna del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]