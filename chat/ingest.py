#!/usr/bin/env python3
import base64
import os
import glob
import logging
import datetime
from pathlib import Path
from typing import List, Any
from multiprocessing import Pool
from tqdm import tqdm

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import CHROMA_SETTINGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import pdfplumber

import nltk
# nltk.download('all') # comment after used
from langchain_core.documents import Document
from typing import List

logging.getLogger("pdfminer").setLevel(logging.ERROR)


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-mpnet-base-v2')
chunk_size = 512  
chunk_overlap = 128  





# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def clean_text(content: str) -> str:
    """Remove excessive whitespace and line breaks"""
    content = ' '.join(content.split())  # Collapse whitespace
    return content.strip()



def process_pdf_with_tables(file_path: str) -> List[Document]:
    """Convert both text and tables to clean text format"""
    documents = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract regular text first
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=clean_text(text),
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        "element_type": "text"
                    }
                ))
            
            # Process all tables into readable text format
            for table_num, table in enumerate(page.extract_tables(), start=1):
                table_str = format_table_as_text(table)
                if table_str:
                    documents.append(Document(
                        page_content=f"TABLE {table_num}:\n{table_str}",
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "element_type": "table",
                            "table_num": table_num
                        }
                    ))
    
    return documents


def format_table_as_text(table_data: List[List[Any]]) -> str:
    """Convert table data to clean, readable text format"""
    if not table_data or len(table_data) < 2:
        return ""
    
    # Determine column widths
    col_widths = []
    for col_idx in range(len(table_data[0])):
        max_len = max(len(str(row[col_idx])) for row in table_data if col_idx < len(row))
        col_widths.append(min(max_len, 30))  # Cap at 30 chars
    
    # Build table string
    table_str = ""
    for row_idx, row in enumerate(table_data):
        # Skip empty rows
        if not any(cell for cell in row):
            continue
            
        # Build row string
        row_str = ""
        for col_idx, cell in enumerate(row):
            if col_idx >= len(col_widths):
                continue
            cell_str = str(cell or "").replace("\n", " ").strip()
            row_str += cell_str.ljust(col_widths[col_idx]) + "  "
        
        table_str += row_str.strip() + "\n"
        
        # Add separator after header
        if row_idx == 0:
            table_str += "-" * len(row_str) + "\n"
    
    return table_str.strip()

def load_single_document(file_path: str) -> List[Document]:
    """Unified document loader with error handling"""
    try:

        if file_path.endswith('.DS_Store'):
            return []
        
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        
        if ext == ".pdf":
            docs = process_pdf_with_tables(file_path)
        elif ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            if loader_class is None:
                return []
            loader = loader_class(file_path, **loader_args)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file extension '{ext}'")
        
        # Add consistent metadata
        for doc in docs:
            doc.metadata.update({
                "file_type": ext[1:],
                "file_path": file_path,
                "loaded_at": datetime.datetime.now().isoformat()
            })
        
        return docs
    
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {str(e)}")
        return []



def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update existing vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        collection = db.get()
        # Get all unique source files from existing collection
        existing_sources = {metadata['source'] for metadata in collection['metadatas']}
        # Process only new documents not already in the vectorstore
        all_files = [str(path) for path in Path(source_directory).rglob("*") 
                    if path.is_file() and not path.name.startswith('.')]  # Skip hidden files
        
        new_files = [file for file in all_files 
                    if file not in existing_sources and not file.endswith('.DS_Store')]
        
        if not new_files:
            print("No new documents to add")
            return
            
        print(f"Found {len(new_files)} new documents to process")
        documents = []
        for file in new_files:
            docs = load_single_document(file)
            documents.extend(docs)
        
        if not documents:
            print("No new documents loaded")
            return
            
        print(f"Creating embeddings for {len(documents)} chunks. May take some minutes...")
        db.add_documents(documents)
    else:
        # Create new vectorstore
        print("Creating new vectorstore")
        documents = []
        all_files = [str(path) for path in Path(source_directory).rglob("*") 
                    if path.is_file() and not path.name.startswith('.')]  # Skip hidden files
        
        for file in all_files:
            if file.endswith('.DS_Store'):  # Skip macOS system files
                continue
            docs = load_single_document(file)
            documents.extend(docs)
        
        if not documents:
            print("No documents found to process")
            return
            
        print(f"Creating embeddings for {len(documents)} chunks. May take some minutes...")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS
        )
    
    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()


