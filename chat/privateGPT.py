#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain.llms import Ollama
from langchain_ollama.llms import OllamaLLM
import chromadb
import os
import argparse
import time

model = os.environ.get("MODEL", "deepseek-r1")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',6))

from constants import CHROMA_SETTINGS


def main():
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    def enhanced_retriever(query: str):
        """Improved retriever that better handles table queries"""
        # Normalize query for table detection
        normalized_query = query.lower()
        table_keywords = [
            "cuántos", "cuántas", "número", "dato", 
            "estadística", "tabla", "porcentaje", "población"
        ]
        
        # Check if query is table-oriented
        is_table_query = any(kw in normalized_query for kw in table_keywords)
        
        # First: Get regular text results
        text_results = db.similarity_search(
            query,
            filter={"element_type": "text"},
            k=target_source_chunks // 2
        )
        
        # Second: Special table handling
        table_results = []
        if is_table_query:
            # Try multiple table-specific queries
            table_queries = [
                f"DATOS NUMÉRICOS SOBRE: {query}",
                f"TABLA ESTADÍSTICA: {query}",
                f"POBLACIÓN: {query}"
            ]
            
            for tq in table_queries:
                table_results.extend(db.similarity_search(
                    tq,
                    filter={"element_type": "table"},
                    k=target_source_chunks
                ))
        
        # Combine and deduplicate
        combined = text_results + table_results
        seen = set()
        final_results = []
        
        for doc in combined:
            if doc.metadata["source"] not in seen:
                seen.add(doc.metadata["source"])
                # Boost table scores for table queries
                if is_table_query and doc.metadata.get("element_type") == "table":
                    doc.metadata["score"] = doc.metadata.get("score", 1) * 1.5
                final_results.append(doc)
        
        return sorted(final_results, 
                    key=lambda x: -x.metadata.get("score", 1))[:target_source_chunks]


    retriever = db.as_retriever(
        search_kwargs={"k": target_source_chunks},
        search_type="similarity",
        search_func=enhanced_retriever
    )

    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = OllamaLLM(
        model=model,
        callbacks=callbacks,
        num_ctx=4096,
    )

    # Enhanced prompt template
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Sigue estos pasos para responder:
        1. Identifica si la pregunta requiere datos de tablas (búsqueda de números, comparaciones, estadísticas)
        2. Para preguntas de tabla:
           a) Localiza todas las tablas relevantes
           b) Extrae valores EXACTOS (ej: "Caldas: 22,659 en 2005")
           c) Realiza cálculos si es necesario
        3. Para texto normal, resume la información clave
        4. Siempre menciona las fuentes utilizadas

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta (en español claro y detallado):"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
        chain_type_kwargs={
            "prompt": qa_prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        }
    )

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        spanish_query = f"{query} \n\nResponde en español de manera clara y detallada."
        start = time.time()
        res = qa.invoke({"query": spanish_query})
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        print("\n\n> Question:")
        print(query)
        print(answer)


        for document in docs:
            print("\n> " + document.metadata["source"] + " (Página " + str(document.metadata.get("page", "N/A")) + ")")
            if document.metadata.get("element_type") == "table":
                print("[TABLA]")
                content = document.page_content
                if "MARKDOWN FORMAT:" in content:
                    print(content.split("MARKDOWN FORMAT:")[1].strip())
                else:
                    print(content)
            else:
                print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
        'using the power of LLMs.'
    )
    parser.add_argument(
        "--hide-source", "-S",
        action='store_true',
        help='Use this flag to disable printing of source documents used for answers.'
    )
    parser.add_argument(
        "--mute-stream", "-M",
        action='store_true',
        help='Use this flag to disable the streaming StdOut callback for LLMs.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()
