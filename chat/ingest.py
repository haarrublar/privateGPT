#!/usr/bin/env python3
import base64
import os
import json
import glob
import logging
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
# from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import pdfplumber
from unstructured.partition.pdf import partition_pdf
import nltk
# nltk.download('all') # comment after used
from langchain_core.documents import Document
from typing import List
logging.getLogger("pdfminer").setLevel(logging.ERROR)


# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 1000
chunk_overlap = 200

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    # ".pdf": (PyMuPDFLoader, {}),
    ".pdf": (None, {}), 
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

# def clean_text(content: str) -> str:
#     """Remove excessive whitespace and line breaks"""
#     content = ' '.join(content.split())  # Collapse whitespace
#     return content.strip()

def clean_text(content: str) -> str:
    """Preserve table formatting while cleaning other text"""
    if content.startswith("TABLE") and "\n|" in content:
        return content  # Don't modify tables
    return ' '.join(content.split()).strip()



# def load_single_document(file_path: str) -> List[Document]:
#     ext = "." + file_path.rsplit(".", 1)[-1].lower()
    
#     if ext == ".pdf":
#         return process_pdf_with_tables(file_path)
#     elif ext in LOADER_MAPPING:
#         loader_class, loader_args = LOADER_MAPPING[ext]
#         loader = loader_class(file_path, **loader_args)
#         return loader.load()
    
#     raise ValueError(f"Unsupported file extension '{ext}'")

# def process_pdf_with_tables(file_path: str) -> List[Document]:
    
#     documents = []
#     with pdfplumber.open(file_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             # Text extraction
#             text = page.extract_text()
#             if text and text.strip():
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={
#                         "source": file_path,
#                         "page": page_num + 1,
#                         "element_type": "text"
#                     }
#                 ))
            
#             # Table extraction
#             for table_num, table in enumerate(page.extract_tables()):
#                 table_text = ""
#                 for row in table:
#                     cleaned_row = [str(cell or "") for cell in row]
#                     table_text += " | ".join(cleaned_row) + "\n"
                
#                 if table_text.strip():
#                     documents.append(Document(
#                         page_content=f"Table {table_num+1}:\n{table_text.strip()}",
#                         metadata={
#                             "source": file_path,
#                             "page": page_num + 1,
#                             "element_type": "table"
#                         }
#                     ))
    
#     return documents

# def process_pdf_with_tables(file_path: str) -> List[Document]:
#     """Process PDF with pdfplumber (tables + text)"""
#     documents = []
    
#     try:
#         with pdfplumber.open(file_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 # Text extraction with cleaning
#                 text = page.extract_text()
#                 if text and text.strip():
#                     documents.append(Document(
#                         page_content=clean_text(text),
#                         metadata={
#                             "source": file_path,
#                             "page": page_num,
#                             "element_type": "text"
#                         }
#                     ))
                
#                 # Robust table extraction
#                 try:
#                     for table_num, table in enumerate(page.extract_tables(), start=1):
#                         # Handle empty cells and None values
#                         table_rows = []
#                         for row in table:
#                             cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
#                             table_rows.append(" | ".join(cleaned_row))
                        
#                         if any(row.strip() for row in table_rows):
#                             documents.append(Document(
#                                 page_content=f"TABLE {table_num}:\n" + "\n".join(table_rows),
#                                 metadata={
#                                     "source": file_path,
#                                     "page": page_num,
#                                     "element_type": "table",
#                                     "table_num": table_num
#                                 }
#                             ))
#                 except Exception as table_error:
#                     print(f"Table extraction error in {file_path} page {page_num}: {table_error}")
                    
#     except Exception as pdf_error:
#         print(f"PDF processing failed for {file_path}: {pdf_error}")
    
#     return documents

# def summarize_table_content(table_data):
#     headers = table_data[0]
#     sample_data = []
#     for row in table_data[1:4]:  # First few data rows
#         sample_data.append(", ".join(f"{headers[i]}: {cell}" 
#                       for i, cell in enumerate(row) if i < len(headers)))
#     return "Contains data about: " + "; ".join(sample_data)

# def process_pdf_with_tables(file_path: str) -> List[Document]:
#     """Process PDF with enhanced progress tracking"""
#     documents = []
#     pdf = None
    
#     try:
#         print(f"📂 Loading {os.path.basename(file_path)}...")
#         pdf = pdfplumber.open(file_path)
#         total_pages = len(pdf.pages)
        
#         for page_num, page in enumerate(pdf.pages, start=1):
#             page_docs = []
            
#             # Text extraction
#             try:
#                 print(f"📄 Page {page_num}/{total_pages} - Extracting text...", end="\r")
#                 text = page.extract_text()
#                 if text and text.strip():
#                     page_docs.append(Document(
#                         page_content=clean_text(text),
#                         metadata={"source": file_path, "page": page_num, "element_type": "text"}
#                     ))
#             except Exception as e:
#                 print(f"\n⚠️ Text error on page {page_num}: {str(e)}")
            
#             # Table extraction
#             try:
#                 tables = page.find_tables()
#                 if tables:
#                     print(f"📊 Page {page_num} - Found {len(tables)} tables...", end="\r")
#                     for table_num, table in enumerate(tables, start=1):
#                         try:
#                             table_data = table.extract()
#                             if table_data:
#                                 # Build markdown table
#                                 rows = [
#                                     "| " + " | ".join(
#                                         str(cell or "").replace("\n", " ").strip()
#                                         for cell in row
#                                     ) + " |"
#                                     for row in table_data
#                                 ]
                                
#                                 if rows:
#                                     # Add header separator
#                                     separator = "|" + "|".join(["---"] * len(table_data[0])) + "|"
#                                     rows.insert(1, separator)
                                    
#                                     # Create the Document object
#                                 table_description = (
#                                     f"Table {table_num} with {len(table_data[0])} columns and {len(table_data)} rows. "
#                                     f"Column headers: {', '.join(str(cell) for cell in table_data[0])}. "
#                                     "Content summary: " + summarize_table_content(table_data)  # Implement this helper
#                                 )

#                                 table_doc = Document(
#                                     page_content=f"TABLE {table_num}:\n{table_description}\n\nRaw data:\n" + "\n".join(rows),
#                                     metadata={
#                                         "source": file_path,
#                                         "page": page_num,
#                                         "element_type": "table",
#                                         "table_num": table_num,
#                                         "table_format": "markdown",
#                                         "columns": len(table_data[0]),
#                                         "rows": len(table_data)
#                                     }
#                                 )


#                                 page_docs.append(table_doc)
#                         except Exception as e:
#                             print(f"\n⚠️ Table {table_num} error: {str(e)}")
#             except Exception as e:
#                 print(f"\n⚠️ Table finding error: {str(e)}")
            
#             if page_docs:
#                 documents.extend(page_docs)
#                 print(f"✅ Page {page_num} - Added {len(page_docs)} elements".ljust(80))
#             else:
#                 print(f"❌ Page {page_num} - No content extracted".ljust(80))
                
#     except Exception as e:
#         print(f"\n🔥 Failed to process {file_path}: {str(e)}")
#         return []
    
#     finally:
#         if pdf:
#             pdf.close()
#             print(f"🗂️ Closed {os.path.basename(file_path)}")
    
#     print(f"\n🎉 Processed {len(documents)} elements from {file_path}")
#     return documents


def summarize_table_content(table_data: List[List[str]]) -> dict:
    """Lightweight table analysis returning JSON-serializable stats"""
    headers = table_data[0]
    analysis = {
        "column_types": {},
        "numeric_ranges": {},
        "value_samples": {}
    }

    for col_idx, header in enumerate(headers):
        values = []
        numeric_values = []
        
        # Scan first 10 rows for column analysis
        for row in table_data[1:11]:  # Skip header, limit to 10 rows
            if col_idx >= len(row):
                continue
                
            value = str(row[col_idx]).strip() if row[col_idx] is not None else ""
            if value:
                values.append(value)
                if value.replace('.', '', 1).isdigit():
                    numeric_values.append(float(value) if '.' in value else int(value))

        # Store column properties
        analysis['value_samples'][header] = list(set(values))[:3]  # Unique samples
        
        if numeric_values:
            analysis['column_types'][header] = "numeric"
            analysis['numeric_ranges'][header] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "avg": sum(numeric_values)/len(numeric_values)
            }
        else:
            analysis['column_types'][header] = "text"

    return analysis


def process_table(table_data: List[List[Any]], table_num: int, page_num: int, file_path: str) -> Document:
    """Optimized table processing with dual representations"""
    try:
        # 1. Clean and validate data
        headers = [str(cell).strip() if cell is not None else "" 
                  for cell in table_data[0]]
        rows = table_data[1:]
        
        # 2. Create multiple representations
        # Structured representation (for retrieval)
        structured_data = []
        for row in rows:
            structured_row = {}
            for i, cell in enumerate(row):
                if i < len(headers):
                    cell_value = str(cell).strip() if cell is not None else ""
                    structured_row[headers[i]] = cell_value
            structured_data.append(structured_row)
        
        # Flattened text representation (for embedding)
        flattened_text = " | ".join(headers) + "\n"
        for row in structured_data:
            flattened_text += " | ".join(row.values()) + "\n"
        
        # 3. Enhanced metadata
        metadata = {
            "source": file_path,
            "page": page_num,
            "element_type": "table",
            "table_id": f"{os.path.basename(file_path)}_p{page_num}_t{table_num}",
            "columns": headers,
            "num_rows": len(rows),
            "data_type": "structured",
            "contains_numerics": any(
                any(cell.replace('.','',1).isdigit() 
                for cell in row.values())
                for row in structured_data
            )
        }
        
        # 4. Create document with both representations
        return Document(
            page_content=(
                f"TABLE {table_num} STRUCTURED:\n{json.dumps(structured_data, ensure_ascii=False)}\n\n"
                f"TABLE {table_num} FLATTENED:\n{flattened_text}"
            ),
            metadata=metadata
        )
    
    except Exception as e:
        print(f"\n⚠️ Table processing error (Page {page_num}, Table {table_num}): {str(e)}")
        return None


def process_pdf_with_tables(file_path: str) -> List[Document]:
    """Robust PDF processor with comprehensive error handling"""
    documents = []
    pdf = None
    
    def validate_metadata(metadata: dict) -> bool:
        """Ensure metadata is ChromaDB-compatible with JSON fallback"""
        try:
            allowed_types = (str, int, float, bool, type(None))
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = json.dumps(value, ensure_ascii=False)
                elif not isinstance(value, allowed_types):
                    return False
            return True
        except Exception as e:
            print(f"Metadata validation error: {str(e)}")
            return False

    try:
        print(f"📂 Loading {os.path.basename(file_path)}...")
        pdf = pdfplumber.open(file_path)
        total_pages = len(pdf.pages)
        
        for page_num, page in enumerate(pdf.pages, start=1):
            page_docs = []
            
            # Text extraction - nested try-except-finally
            try:
                print(f"📄 Page {page_num}/{total_pages} - Extracting text...", end="\r")
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        page_docs.append(Document(
                            page_content=clean_text(text),
                            metadata={
                                "source": file_path, 
                                "page": page_num, 
                                "element_type": "text"
                            }
                        ))
                except Exception as e:
                    print(f"\n⚠️ Text extraction error on page {page_num}: {str(e)}")
                finally:
                    # Cleanup text extraction resources if needed
                    pass
                
            except Exception as e:
                print(f"\n⚠️ Outer text processing error on page {page_num}: {str(e)}")
            
            # Table processing - nested try-except-finally
            try:
                tables = page.find_tables()
                if tables:
                    print(f"📊 Page {page_num} - Found {len(tables)} tables...", end="\r")
                    for table_num, table in enumerate(tables, start=1):
                        try:
                            raw_data = table.extract()
                            if not raw_data or len(raw_data) < 2:
                                continue
                                
                            table_doc = process_table(raw_data, table_num, page_num, file_path)
                            if table_doc and validate_metadata(table_doc.metadata):
                                page_docs.append(table_doc)
                                
                        except Exception as e:
                            print(f"\n⚠️ Table {table_num} processing error: {str(e)}")
                        finally:
                            # Ensure resources are cleaned up per table
                            if 'raw_data' in locals():
                                del raw_data
                
            except Exception as e:
                print(f"\n⚠️ Table finding error on page {page_num}: {str(e)}")
            finally:
                # Ensure page resources are released
                if 'page' in locals():
                    del page
            
            if page_docs:
                documents.extend(page_docs)
                print(f"✅ Page {page_num} - Added {len(page_docs)} elements".ljust(80))
                
    except Exception as e:
        print(f"\n🔥 Fatal processing error: {str(e)}")
        return []
    finally:
        if pdf:
            try:
                pdf.close()
                print(f"🗂️ Closed {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️ File closure error: {str(e)}")
            finally:
                # Final PDF cleanup
                if 'pdf' in locals():
                    del pdf

    # Final reporting
    try:
        table_count = sum(1 for d in documents if d.metadata.get("element_type") == "table")
        numeric_tables = sum(1 for d in documents if d.metadata.get("has_numerics", False))
        
        print(f"\n📊 Extraction Complete:")
        print(f"- Pages processed: {total_pages}")
        print(f"- Tables extracted: {table_count} ({numeric_tables} with numeric data)")
        print(f"- Total elements: {len(documents)}")
    except Exception as e:
        print(f"\n⚠️ Final reporting error: {str(e)}")
    finally:
        # Final document cleanup if needed
        pass
    
    return documents


def load_single_document(file_path: str) -> List[Document]:
    """Robust document loader with table validation"""
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    try:
        if ext == ".pdf":
            docs = process_pdf_with_tables(file_path)
            valid_docs = []
            
            for doc in docs:
                try:
                    # Ensure basic metadata
                    doc.metadata.update({
                        "file_type": ext[1:],
                        "file_path": file_path
                    })
                    
                    # Special handling for tables
                    if doc.metadata.get("element_type") == "table":
                        content = doc.page_content
                        
                        # Check for both possible table formats
                        if content.startswith(('{', '[')):
                            try:
                                # Validate and normalize JSON
                                table_data = json.loads(content)
                                doc.page_content = json.dumps(table_data, ensure_ascii=False)
                            except json.JSONDecodeError:
                                # Fallback to markdown if JSON invalid
                                if "|" in content:  # Markdown table detected
                                    doc.metadata["table_format"] = "markdown"
                                else:
                                    print(f"⚠️ Invalid table format in {file_path}, keeping raw content")
                                    doc.page_content = f"TABLE DATA:\n{content}"
                        else:
                            # Handle non-JSON tables (markdown or raw)
                            doc.metadata["table_format"] = "markdown" if "|" in content else "raw"
                    
                    valid_docs.append(doc)
                    
                except Exception as doc_error:
                    print(f"⚠️ Document processing error in {file_path}: {str(doc_error)}")
                    continue
                    
            return valid_docs

        elif ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            if loader_class is None:
                return []
            
            try:
                loader = loader_class(file_path, **loader_args)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "file_type": ext[1:],
                        "file_path": file_path
                    })
                return docs
            except Exception as loader_error:
                print(f"⚠️ Loader error for {file_path}: {str(loader_error)}")
                return []

        else:
            raise ValueError(f"Unsupported file extension '{ext}'")

    except Exception as e:
        print(f"⛔ Critical error loading {file_path}: {str(e)}")
        return []


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """Robust parallel document loader"""
    # Validate input directory
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Gather files with error handling
    all_files = []
    try:
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
    except Exception as glob_error:
        print(f"⚠️ File discovery error: {str(glob_error)}")
        return []

    # Filter files safely
    filtered_files = []
    for file_path in all_files:
        try:
            if file_path not in ignored_files:
                filtered_files.append(file_path)
        except Exception as filter_error:
            print(f"⚠️ File filtering error for {file_path}: {str(filter_error)}")

    # Parallel loading with enhanced error handling
    results = []
    try:
        with Pool(processes=min(4, os.cpu_count())) as pool:  # Conservative pool size
            with tqdm(total=len(filtered_files), desc='Loading documents', ncols=80) as pbar:
                for docs in pool.imap_unordered(load_single_document, filtered_files):
                    try:
                        if docs:  # Only extend if valid docs
                            results.extend(docs)
                    except Exception as extend_error:
                        print(f"⚠️ Result processing error: {str(extend_error)}")
                    finally:
                        pbar.update()
    except Exception as pool_error:
        print(f"⛔ Parallel processing failed: {str(pool_error)}")
        # Fallback to sequential loading
        results = []
        for file_path in tqdm(filtered_files, desc='Loading (sequential)'):
            try:
                docs = load_single_document(file_path)
                if docs:
                    results.extend(docs)
            except Exception as seq_error:
                print(f"⚠️ Sequential load failed for {file_path}: {str(seq_error)}")

    # Final filtering and validation
    valid_docs = []
    for doc in results:
        try:
            if doc.page_content.strip():
                # Ensure all docs have required metadata
                if "file_type" not in doc.metadata:
                    doc.metadata["file_type"] = "unknown"
                valid_docs.append(doc)
        except Exception as doc_error:
            print(f"⚠️ Document validation error: {str(doc_error)}")

    print(f"\n🎯 Loaded {len(valid_docs)} valid documents from {len(filtered_files)} files")
    return valid_docs


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """Enhanced document processor with table-aware splitting"""
    try:
        print(f"\n🔍 Loading documents from {source_directory}")
        documents = load_documents(source_directory, ignored_files)
        
        if not documents:
            print("⛔ No valid documents found")
            return []

        # Detailed table analysis
        table_docs = [d for d in documents if d.metadata.get("element_type") == "table"]
        table_count = len(table_docs)
        num_numeric_tables = sum(1 for d in table_docs if d.metadata.get("has_numerics", False))
        
        print(f"📊 Loaded {len(documents)} documents ({table_count} tables)")
        print(f"   - Tables with numeric data: {num_numeric_tables}")
        print(f"   - Text documents: {len(documents) - table_count}")

        # Configure splitting strategy
        def split_logic(text: str) -> bool:
            """Determine if text should be split"""
            if text.startswith(('{TABLE', '{"headers"')):  # JSON table markers
                return False
            return None  # Default splitting behavior

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=split_logic,
            keep_separator=True
        )

        # Process documents with error handling
        chunks = []
        chunk_errors = 0
        
        for doc in documents:
            try:
                if doc.metadata.get("element_type") == "table":
                    # Preserve complete tables
                    chunks.append(doc)
                else:
                    # Split regular text
                    chunks.extend(text_splitter.split_documents([doc]))
            except Exception as e:
                print(f"⚠️ Chunking error for {doc.metadata.get('source', 'unknown')}: {str(e)}")
                chunk_errors += 1

        # Filter and validate chunks
        valid_chunks = []
        for chunk in chunks:
            try:
                content = chunk.page_content.strip()
                if content:
                    # Ensure all chunks have required metadata
                    if "source" not in chunk.metadata:
                        chunk.metadata["source"] = "unknown"
                    valid_chunks.append(chunk)
            except Exception as e:
                print(f"⚠️ Invalid chunk: {str(e)}")
                chunk_errors += 1

        # Final reporting
        table_chunks = sum(1 for c in valid_chunks 
                          if c.metadata.get("element_type") == "table")
        
        print(f"\n✂️  Split into {len(valid_chunks)} chunks")
        print(f"   - Table chunks preserved: {table_chunks}")
        print(f"   - Text chunks: {len(valid_chunks) - table_chunks}")
        if chunk_errors > 0:
            print(f"   ⚠️  Errors encountered: {chunk_errors}")

        if not valid_chunks:
            print("⛔ No valid chunks generated")
            return []

        return valid_chunks

    except Exception as e:
        print(f"🔥 Critical processing error: {str(e)}")
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
    # Initialize embeddings with explicit parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if does_vectorstore_exist(persist_directory):
        # Update existing vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        
        # Get existing sources while preserving table documents
        existing_sources = set()
        try:
            # Chroma >= 0.4.0 style
            collection_data = db._collection.get()
            for metadata in collection_data['metadatas']:
                if metadata and metadata.get('element_type') != 'table':
                    existing_sources.add(metadata['source'])
        except AttributeError:
            # Fallback for older Chroma versions
            collection_data = db.get()
            for metadata in collection_data['metadatas']:
                if metadata and metadata.get('element_type') != 'table':
                    existing_sources.add(metadata['source'])
        
        texts = process_documents(list(existing_sources))
        print(f"Adding {len(texts)} new documents (including table updates)...")
        db.add_documents(texts)
    else:
        # Create new vectorstore
        print("Creating new vectorstore with table support")
        texts = process_documents()
        
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
            collection_metadata={
                "hnsw:space": "cosine",
                "table_support": "enhanced",
                "content_types": "text,table"
            }
        )
    
    # Post-ingestion validation with version-agnostic approach
    try:
        # Chroma >= 0.4.0
        collection = db._collection
        collection_data = collection.get()
        table_count = sum(1 for m in collection_data['metadatas'] if m and m.get('element_type') == 'table')
        total_count = collection.count()
        embedding_dim = len(collection_data['embeddings'][0]) if collection_data['embeddings'] else 'N/A'
    except AttributeError:
        # Fallback for older versions
        collection_data = db.get()
        table_count = sum(1 for m in collection_data['metadatas'] if m and m.get('element_type') == 'table')
        total_count = len(collection_data['ids'])
        embedding_dim = 'N/A'
    
    print(f"\nIngestion complete! Final stats:")
    print(f"- Total documents: {total_count}")
    print(f"- Tables stored: {table_count}")
    print(f"- Embedding dimension: {embedding_dim}")
    print("\nYou can now query your documents including table data.")

if __name__ == "__main__":
    main()

