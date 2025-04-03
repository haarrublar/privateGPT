import requests
import pandas as pd
import json
from typing import Any, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo


def query_ollama_model(model: str = "llama3.2",
                       stream: bool = False,
                       messages: Optional[Dict[str, Any]] = None, 
                       url: str = "http://localhost:11434/api/chat",
                       **kwargs
                    ):
    url = url
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs
    }

    agent_response = requests.post(url, json=payload)

    return agent_response


def generate_aiollama_response(
        model: str, 
        doc: str, 
        query: str,
        stream: bool = False, 
        **kwargs
    ):

    est = datetime.now(ZoneInfo("America/New_York"))
    timestamp = est.strftime("%Y-%m-%d-%Hhr-%Mmin")

    responses_path = f"./responses/{timestamp}.json"
    load_data = pd.read_csv(doc, delimiter='\t')
    contextual_query = """Revisa cuidadosamente el texto proporcionado y extrae los aspectos clave. Concéntrate en los siguientes puntos:
    1. "Números" – Identifica y enumera los valores numéricos relevantes, porcentajes o cualquier cifra estadística.
    2. "Estadísticas" – Resume los hallazgos estadísticos, tendencias o métricas clave mencionadas.
    3. "Metodologías" – Describe los métodos, técnicas o enfoques utilizados.
    4. "Contexto" – Explica el trasfondo, el entorno o las circunstancias relacionadas con los datos.
    5. "Objetivo principal" – Indica claramente el propósito u objetivo principal presentado en el texto.
    """
    technical_query = """Analiza cuidadosamente el texto proporcionado y extrae información clave con un enfoque estrictamente técnico. Concéntrate en los siguientes aspectos:

    1. **Datos y estructura** – Describe la naturaleza de los datos, su formato, tipos de variables (categóricas, numéricas, ordinales), cantidad de registros y cualquier aspecto relevante sobre su organización o distribución.

    2. **Análisis estadístico** – Extrae y resume los indicadores estadísticos clave, incluyendo medidas de tendencia central (media, mediana, moda), dispersión (varianza, desviación estándar, rango intercuartílico), correlaciones y cualquier prueba estadística aplicada.

    3. **Modelado matemático** – Describe los modelos matemáticos utilizados, como regresiones, optimización, ecuaciones diferenciales, series temporales u otras formulaciones.

    4. **Métodos computacionales** – Especifica los algoritmos, técnicas de procesamiento de datos y herramientas computacionales empleadas, incluyendo machine learning, deep learning, optimización numérica o simulaciones.

    5. **Inferencias y validación** – Explica cómo se han realizado inferencias a partir de los datos, qué técnicas de validación o métricas de evaluación se han empleado (como error cuadrático medio, precisión, recall, AUC-ROC) y si se han aplicado métodos de validación cruzada.

    6. **Implementación y rendimiento** – Detalla aspectos relacionados con la eficiencia computacional, tiempo de ejecución, complejidad algorítmica y escalabilidad de los métodos empleados.

    """


    instruction = f"cuantas personas afrocolombianas hay en caldas segun el texto\n```\n{str(load_data)}\n```"

    messages = [
        {"role": "system", "content": "Eres un asistente investigativo estadistico con enfoque social. Tu tarea es como investigador extraer información de manera precisa minuciosa y detallada. Responde completamente y sin omitir ninguna instrucción."},
        {"role": "user", "content": instruction}
    ]

    aiagent_response = query_ollama_model(model=model, 
                                          stream=stream,
                                          messages=messages,
                                          **kwargs
                                        )


    with open(responses_path, 'w', encoding='utf-8') as document:
        json.dump(aiagent_response.json(), document, ensure_ascii=False, indent=4)
    

generate_aiollama_response(
    model='llama3.2',
    doc='kmeans_tesis_method.txt',
    query='contextual_query'
)


