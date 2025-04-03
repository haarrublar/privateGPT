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

    def table_aware_retriever(query: str):
        # First get regular results
        regular_results = db.similarity_search(query, k=target_source_chunks)
        
        # Then search specifically for tables
        table_results = db.similarity_search(
            f"TABLE CONTENT RELATED TO: {query}",
            filter={"element_type": "table"},
            k=max(1, target_source_chunks//2)
        )
        
        # Combine and deduplicate
        combined = regular_results + table_results
        seen = set()
        unique_results = []
        for doc in combined:
            if doc.metadata["source"] not in seen:
                seen.add(doc.metadata["source"])
                unique_results.append(doc)
        return unique_results[:target_source_chunks]

    retriever = db.as_retriever(
        search_kwargs={"k": target_source_chunks},
        search_type="similarity",
        search_func=table_aware_retriever
    )

    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = OllamaLLM(
        model=model,
        callbacks=callbacks,
        num_ctx=4096,
    )

    # Custom prompt that handles tables better
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Analyze the following context which may include text and tables to answer the question.
        Pay special attention to tables as they may contain important numerical data.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in clear Spanish, mentioning if any tables were used in deriving the answer."""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
        chain_type_kwargs={"prompt": qa_prompt}
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
            print("\n> " + document.metadata["source"] + ":")
            if document.metadata.get("element_type") == "table":
                print(f"[TABLE FROM PAGE {document.metadata['page']}]")
                # Extract just the markdown table if available
                if "Raw data:" in document.page_content:
                    print(document.page_content.split("Raw data:")[-1].strip())
                else:
                    print(document.page_content)
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
