#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
# from langchain.llms import Ollama
from langchain_ollama.llms import OllamaLLM
import chromadb
import os
import argparse
import time
from typing import List
from langchain.schema import Document

model = os.environ.get("MODEL", "deepseek-r1")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',6))

from constants import CHROMA_SETTINGS



def main():
    # Parse the command line arguments
    args = parse_arguments()
    
    # Enhanced embeddings configuration
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        encode_kwargs={'normalize_embeddings': True}
    )

    # Improved ChromaDB configuration with client settings
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    # Enhanced retriever with numerical data prioritization
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": target_source_chunks + 2,
            "score_threshold": 0.25,
            "fetch_k": target_source_chunks * 3,
            "filter": {
                "$or": [
                    {"element_type": "table"},
                    {"element_type": "figure"},
                    {"file_type": "pdf"}
                ]
            }
        }
    )

    # Custom prompt template for better numerical responses
    custom_prompt = PromptTemplate(
        template="""Contexto: {context}

Pregunta: {question}

Responde en español siguiendo estas reglas:
1. Si hay datos numéricos, cita exactamente: "Según [documento] (página X)..."
2. Para tablas/figuras: "La Tabla Y muestra que..."
3. Si no hay datos exactos, explica qué información relacionada existe
4. Nunca inventes números

Ejemplo bueno:
"Según el documento X (página 5, Figura 3), hay 2,500 afrocolombianos en Caldas."

Ahora responde:""",
        input_variables=["context", "question"]
    )

    # Configure LLM and QA chain
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    llm = OllamaLLM(
        model=model,
        callbacks=callbacks,
        num_ctx=4096,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=not args.hide_source
    )

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Enhance numerical queries
        enhanced_query = query
        # In your interactive loop:
        numerical_keywords = ["cuántos", "cuántas", "número", "cantidad", "estadística", "porcentaje", "población"]
        if any(keyword in query.lower() for keyword in numerical_keywords):
            enhanced_query = (
                f"{query} Proporciona TODOS los datos numéricos relevantes de tablas o figuras. "
                "Si hay múltiples fuentes, compara los datos. Si no hay datos exactos, explica por qué."
            )
        else:
            enhanced_query = query
        if any(keyword in query.lower() for keyword in numerical_keywords):
            enhanced_query = f"{query} Proporciona datos exactos de tablas o figuras si están disponibles."

        start = time.time()
        res = qa.invoke({"query": enhanced_query})
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Verify numerical data was included
        answer = verify_answer(answer, docs)  # Implement this function (shown below)

        # Print results
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Enhanced source display
        if docs:
            print("\n> Fuentes consultadas:")
            for i, document in enumerate(docs, 1):
                print(f"\n[{i}] {document.metadata['source']}")
                if "page" in document.metadata:
                    print(f"   Página: {document.metadata['page']}")
                if "element_type" in document.metadata:
                    print(f"   Tipo: {document.metadata['element_type'].upper()}")
                
                # Show concise preview
                preview = (document.page_content[:150] + "...") if len(document.page_content) > 150 else document.page_content
                print(f"   Contenido relevante: {preview}")

        print(f"\nTiempo de respuesta: {end - start:.2f}s")

def verify_answer(answer: str, sources: List[Document]) -> str:
    """Enhanced verification with table-specific checks"""
    # Check for tables in sources
    table_data = []
    for doc in sources:
        if doc.metadata.get("element_type") == "table":
            table_data.append(f"\n- Tabla en {doc.metadata.get('source', 'documento')}")
            if "page" in doc.metadata:
                table_data[-1] += f" (página {doc.metadata['page']})"
            table_data[-1] += f":\n{doc.page_content[:500]}..."  # Show table preview
    
    if table_data and "Tabla" not in answer:
        table_note = "\n\n[ATENCIÓN: Se encontraron estas tablas relevantes no mencionadas:]"
        table_note += "\n".join(table_data[:3])  # Show max 3 tables
        return answer + table_note
    
    return answer


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
