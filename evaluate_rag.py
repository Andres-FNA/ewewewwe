# ============================================================
# evaluate_rag.py
# Evaluación RAG con Ollama + RAGAS
# Juez: Ollama local (sin OpenAI)
# Sin tabla fancy_grid, salida limpia con to_string()
# ============================================================

import pandas as pd
from tabulate import tabulate

# ============================================================
# RAGAS
# ============================================================

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# Wrappers Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ============================================================
# TU SISTEMA RAG
# ============================================================

from vector_store import VectorStore, EMBEDDING_MODEL
from rag_engine import RAGEngine, TOP_K
from main import INDEX_FOLDER, CHUNK_SIZE, OVERLAP


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

LLM_GENERADOR = "mistral"
LLM_JUEZ = "mistral"
EMBEDDING_JUDGE = "nomic-embed-text"

CHUNK_OVERLAP = OVERLAP

print("\n[CONFIG] Sistema de evaluación iniciado")
print(f"LLM generador: {LLM_GENERADOR}")
print(f"LLM juez: {LLM_JUEZ}")
print(f"Embedding judge: {EMBEDDING_JUDGE}")


# ============================================================
# OLLAMA COMO JUEZ
# ============================================================

print("\nConfigurando juez con Ollama...")

llm_juez = LangchainLLMWrapper(
    ChatOllama(
        model=LLM_JUEZ,
        temperature=0
    )
)

embeddings_juez = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(
        model=EMBEDDING_JUDGE
    )
)

print("[OK] Juez configurado correctamente")


# ============================================================
# CASOS DE PRUEBA
# ============================================================

muestras_evaluacion = [

    # --------------------------------------------------------
    # LITERAL
    # --------------------------------------------------------

    {
        "tipo": "Literal",
        "user_input": "¿Cual es el objetivo de la asignatura de Ingeniería de Software I?",
        "reference": "Ingenieria de Software I",
    },

    {
        "tipo": "Literal",
        "user_input": "Que ruta formativa tiene la asignatura de Ingeniería de Software I?",
        "reference": "Analisis y diseño de sistemas",
    },

    # --------------------------------------------------------
    # SEMÁNTICO
    # --------------------------------------------------------

   {
    "tipo": "Semántico",
    "user_input": "¿Cual es la finalidad de la asignatura de Redes de comunicación I?",
    "reference": (
        "Introducir al estudiante en los conceptos y tecnologías "
        "básicas de las redes de telecomunicaciones, incluyendo "
        "modelos OSI, TCP/IP, topologías de red y configuración "
        "de redes de comunicación."
    ),
},

    {
    "tipo": "Semántico",
    "user_input": "¿Porque se deberia ver la asignatura de Ingenieria de Software I?",
    "reference": (
        "Porque permite al estudiante conocer metodologías, "
        "procesos y herramientas de buenas prácticas de "
        "Ingeniería de Software para participar en proyectos "
        "de desarrollo de software."
    ),
    },
    

    # --------------------------------------------------------
    # MULTI-CHUNK
    # --------------------------------------------------------

    {
        "tipo": "Multi-chunk",
        "user_input": "¿Cuál es el objetivo de la asignatura de Ingenieria de Software I y cuáles son sus resultados esperados de aprendizaje?",
        "reference": (
            "Presentar metodologías, procesos y herramientas de buenas "
            "prácticas de Ingeniería de Software y preparar al estudiante "
            "para participar en proyectos de desarrollo de software."
        ),
    },

   {
    "tipo": "Multi-chunk",
    "user_input": "¿Qué metodologías didácticas y estrategias de evaluación utiliza la asignatura de Redes de comunicación I?",
    "reference": (
        "Utiliza aprendizaje basado en problemas, talleres, "
        "laboratorios, quices teórico-prácticos y evaluación "
        "continua mediante actividades prácticas."
    ),
    },

    # --------------------------------------------------------
    # ALUCINACIÓN
    # --------------------------------------------------------

    {
        "tipo": "Alucinación",
        "user_input": "¿Cuánto cuesta la matrícula del semestre en la universidad?",
        "reference": "No encontré información sobre esto en los documentos disponibles.",
    },

    {
        "tipo": "Alucinación",
        "user_input": "¿Cuál es el correo electrónico del profesor de Ingeniería de Software?",
        "reference": "No encontré información sobre esto en los documentos disponibles.",
    },
]

print(f"\nCasos de prueba cargados: {len(muestras_evaluacion)}")


# ============================================================
# CARGAR VECTOR STORE + ENGINE
# ============================================================

print("\nCargando base vectorial...")

store = VectorStore()
store.load(INDEX_FOLDER)

engine = RAGEngine(
    vector_store=store,
    model=LLM_GENERADOR,
    top_k=TOP_K
)

print("[OK] Sistema RAG listo")


# ============================================================
# EJECUTAR PIPELINE
# ============================================================

print("\nEjecutando evaluación...\n")

registros = []

for muestra in muestras_evaluacion:
    pregunta = muestra["user_input"]

    resultado = engine.query(pregunta)

    respuesta_generada = (
        resultado["answer"]
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .strip()
    )

    registros.append({
        "tipo": muestra["tipo"],
        "user_input": pregunta,
        "retrieved_contexts": [
            entry["text"]
            for entry, score in resultado["retrieved_chunks"]
        ],
        "response": respuesta_generada,
        "reference": muestra["reference"],
    })

    print(f"[OK] {pregunta}")
    print(f"→ {respuesta_generada[:150]}\n")


# ============================================================
# DATASET RAGAS
# ============================================================

samples = [
    SingleTurnSample(
        user_input=r["user_input"],
        retrieved_contexts=r["retrieved_contexts"],
        response=r["response"],
        reference=r["reference"],
    )
    for r in registros
]

dataset = EvaluationDataset(samples=samples)


# ============================================================
# MÉTRICAS
# ============================================================

print("\nConfigurando métricas RAGAS...")

faithfulness_metric = Faithfulness(
    llm=llm_juez
)

answer_relevancy_metric = AnswerRelevancy(
    llm=llm_juez,
    embeddings=embeddings_juez
)

context_precision_metric = ContextPrecision(
    llm=llm_juez
)

print("[OK] Métricas listas")


# ============================================================
# EVALUACIÓN
# ============================================================

print("\n" + "=" * 60)
print("EVALUANDO CON RAGAS")
print("=" * 60)

resultados = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
    ]
)

print("[OK] Evaluación completada")


# ============================================================
# DATAFRAME RESULTADOS
# ============================================================

df = resultados.to_pandas()

df.insert(0, "Tipo", [r["tipo"] for r in registros])
df.insert(1, "Pregunta", [r["user_input"] for r in registros])
df.insert(2, "Respuesta_generada", [r["response"] for r in registros])


# ============================================================
# ANÁLISIS CRÍTICO
# ============================================================

def analisis_critico(faith, ar, cp):
    partes = []

    if pd.isna(faith):
        partes.append("⚠ Sin datos de fidelidad")
    elif faith >= 0.85:
        partes.append("Alta fidelidad")
    elif faith >= 0.50:
        partes.append("Fidelidad media")
    else:
        partes.append("⚠ Riesgo de alucinación")

    if pd.isna(ar):
        partes.append("⚠ Sin datos de relevancia")
    elif ar >= 0.80:
        partes.append("Respuesta relevante")
    elif ar >= 0.50:
        partes.append("Relevancia aceptable")
    else:
        partes.append("⚠ Baja relevancia")

    if pd.isna(cp):
        partes.append("⚠ Sin datos de contexto")
    elif cp >= 0.80:
        partes.append("Chunks precisos")
    elif cp >= 0.50:
        partes.append("Algo de ruido")
    else:
        partes.append("⚠ Contexto deficiente")

    return " | ".join(partes)


df["Análisis"] = df.apply(
    lambda row: analisis_critico(
        row.get("faithfulness", float("nan")),
        row.get("answer_relevancy", float("nan")),
        row.get("context_precision", float("nan")),
    ),
    axis=1
)


# ============================================================
# RESULTADOS FINALES
# ============================================================

tabla_final = df[
    [
        "Pregunta",
        "Respuesta_generada",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "Análisis",
    ]
].copy()

tabla_final.columns = [
    "Pregunta",
    "Respuesta del sistema",
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Análisis",
]


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(
    tabla_final.to_string(
        index=False
    )
)

# ============================================================
# PROMEDIOS GLOBALES
# ============================================================

print("\n" + "=" * 60)
print("PROMEDIOS GLOBALES")
print("=" * 60)

print(
    f"Faithfulness:      {tabla_final['Faithfulness'].mean():.4f}"
)

print(
    f"Answer Relevancy:  {tabla_final['Answer Relevancy'].mean():.4f}"
)

print(
    f"Context Precision: {tabla_final['Context Precision'].mean():.4f}"
)


# ============================================================
# EXPORTAR CSV
# ============================================================

output_csv = "resultados_ragas_final.csv"

tabla_final.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n[GUARDADO] Archivo exportado: {output_csv}")


# ============================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURACIÓN DEL SISTEMA")
print("=" * 60)

config_tabla = [
    ["Embeddings", EMBEDDING_MODEL],
    ["chunk_size", CHUNK_SIZE],
    ["chunk_overlap", CHUNK_OVERLAP],
    ["top_k", TOP_K],
    ["LLM generador", LLM_GENERADOR],
    ["LLM juez", LLM_JUEZ],
]

print(
    tabulate(
        config_tabla,
        headers=["Parámetro", "Valor"],
        tablefmt="fancy_grid"
    )
)