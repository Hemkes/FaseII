# Local AI Helpdesk (Zero Database & Standalone RAG)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<TU-USUARIO>/colab-helpdesk-ai/blob/main/notebooks/helpdesk_rag_colab.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

Solución técnica 100% local, offline y gratuita de Inteligencia Artificial Generativa aplicada a la atención técnica de Helpdesk (Soporte Nivel 1).

Implementa una arquitectura **Zero Database RAG**: la base de conocimientos se vectoriza en memoria RAM mediante álgebra matricial con NumPy y `fastembed`, y la inferencia la ejecuta directamente un modelo cuantizado Llama 3.2 en formato `.gguf` mediante `llama-cpp-python`. No requiere servidores externos ni bases de datos activas en segundo plano.

---

## Características Principales

* **100% Local y Gratuito:** Sin costos de API (cero dependencias de OpenAI o Anthropic).
* **Zero Database:** Indexación y búsqueda semántica por similitud coseno ejecutada completamente en memoria RAM con NumPy.
* **LLM Embebido:** Motor de inferencia integrado en el mismo hilo de ejecución de Python.
* **Dual Run:** Se puede ejecutar localmente mediante Docker o de manera interactiva y acelerada por GPU T4 en Google Colab.
* **API REST Lista:** Interfaz construida sobre FastAPI con esquemas Pydantic y documentación OpenAPI/Swagger automática.

---

## Estructura del Repositorio

```text
colab-helpdesk-ai/
├── data/
│   └── knowledge_base.json         # Base de artículos de soporte
├── models/
│   └── llama-3.2-1b-instruct.Q4_K_M.gguf  # Modelo cuantizado local
├── notebooks/
│   └── helpdesk_rag_colab.ipynb    # Ejecución interactiva en Colab
├── src/
│   ├── __init__.py
│   └── app.py                      # API y motor RAG embebido
├── .dockerignore
├── .env.example
├── Dockerfile
├── README.md
└── requirements.txt