import os
import json
import uuid
from typing import List
from contextlib import asynccontextmanager

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastembed import TextEmbedding
from llama_cpp import Llama

# ---------------------------------------------------------
# Variables de Entorno y Rutas
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/llama-3.2-1b-instruct.Q4_K_M.gguf")
KB_DATA_PATH = os.getenv("KB_DATA_PATH", "data/knowledge_base.json")
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.38"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "2"))

# ---------------------------------------------------------
# Estado en Memoria (Zero Database)
# ---------------------------------------------------------
embed_model = None
llm = None
memory_kb_texts: List[str] = []
memory_kb_vectors: np.ndarray = None


def load_in_memory_kb():
    """Carga knowledge_base.json y genera la matriz vectorial en RAM."""
    global memory_kb_texts, memory_kb_vectors, embed_model
    
    if not os.path.exists(KB_DATA_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de base de conocimientos en: {KB_DATA_PATH}")

    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    memory_kb_texts = [f"{item['topic']}: {item['text']}" for item in docs]
    
    # Genera los embeddings y los guarda como matriz NumPy
    embeddings_generator = embed_model.embed(memory_kb_texts)
    memory_kb_vectors = np.array(list(embeddings_generator))
    print(f"[Embedded RAG] Base de conocimientos cargada: {len(memory_kb_texts)} artículos indexados.")


def search_similar_contexts(query: str) -> List[str]:
    """Calcula similitud coseno en memoria usando álgebra matricial."""
    global memory_kb_texts, memory_kb_vectors, embed_model
    
    if memory_kb_vectors is None or len(memory_kb_texts) == 0:
        return []

    # Vectorizar consulta del usuario
    query_vector = np.array(list(embed_model.embed([query]))[0])

    # Similitud coseno: dot(A, B) / (|A| * |B|)
    norms_kb = np.linalg.norm(memory_kb_vectors, axis=1)
    norm_query = np.linalg.norm(query_vector)

    if norm_query == 0:
        return []

    similarities = np.dot(memory_kb_vectors, query_vector) / (norms_kb * norm_query)

    # Ordenar por mayor puntaje
    ranked_indices = np.argsort(similarities)[::-1][:TOP_K]
    
    return [
        memory_kb_texts[idx]
        for idx in ranked_indices
        if similarities[idx] >= SIMILARITY_THRESHOLD
    ]


# ---------------------------------------------------------
# Ciclo de Vida de FastAPI (Lifespan)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, llm
    
    print("[Startup] Inicializando modelo de embeddings...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print(f"[Startup] Cargando LLM local desde {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {MODEL_PATH}. Verifica la ruta.")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=CPU_THREADS,
        n_gpu_layers=-1 if "COLAB_GPU" in os.environ else 0,  # GPU si detecta Colab, sino CPU
        verbose=False
    )

    print("[Startup] Indexando artículos en memoria...")
    load_in_memory_kb()
    yield
    print("[Shutdown] Liberando recursos...")


# ---------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------
class TicketRequest(BaseModel):
    user_id: str = Field(..., example="usr_102")
    subject: str = Field(..., example="Problema con la VPN")
    description: str = Field(..., example="No puedo conectarme con mi contraseña y token.")

class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    suggested_resolution: str
    context_used: List[str]
    escalation_required: bool

class ReloadResponse(BaseModel):
    status: str
    indexed_articles: int


# ---------------------------------------------------------
# Instancia de FastAPI y Rutas
# ---------------------------------------------------------
app = FastAPI(
    title="Local AI Helpdesk Standalone API",
    description="Helpdesk autónomo con RAG en memoria RAM y LLM embebido.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "in-memory-rag-llm"}

@app.post("/api/v1/kb/reload", response_model=ReloadResponse)
def reload_knowledge_base():
    try:
        load_in_memory_kb()
        return ReloadResponse(status="success", indexed_articles=len(memory_kb_texts))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar base de conocimientos: {str(e)}")

@app.post("/api/v1/tickets/process", response_model=TicketResponse)
def process_ticket(ticket: TicketRequest):
    try:
        full_query = f"{ticket.subject} {ticket.description}"
        contexts = search_similar_contexts(full_query)
        
        # Detección de escalamiento si no hay documentación suficiente
        escalation_required = len(contexts) == 0
        context_str = "\n".join([f"- {c}" for c in contexts]) if contexts else "No hay información documentada para este caso."

        system_instruction = (
            "Eres un agente técnico de Nivel 1 para Helpdesk corporativo.\n"
            "Responde en español con pasos numerados claros, concisos y directos.\n"
            "Si no cuentas con contexto suficiente, sugiere contactar a soporte especializado."
        )

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_instruction}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Información técnica de referencia:
{context_str}

Detalle del problema reportado:
Asunto: {ticket.subject}
Descripción: {ticket.description}

Determina la categoría y los pasos de resolución recomendados:<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        output = llm(
            prompt,
            max_tokens=350,
            temperature=0.1,
            stop=["<|eot_id|>"]
        )

        resolution = output["choices"][0]["text"].strip()

        # Inferencia de categoría
        category = "General"
        for cat in ["VPN", "Contraseña", "Licencias", "Impresora", "Hardware", "Outlook"]:
            if cat.lower() in full_query.lower():
                category = cat
                break

        return TicketResponse(
            ticket_id=f"TICK-{uuid.uuid4().hex[:6].upper()}",
            category=category,
            suggested_resolution=resolution,
            context_used=contexts,
            escalation_required=escalation_required
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el ticket: {str(e)}")