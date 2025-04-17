from langchain_community.document_loaders import WebBaseLoader
import bs4
import pdfplumber
from langchain.schema import Document

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# with pdfplumber.open("./source_documents/LisethBarbosa-TesisMaster.pdf") as pdf:
#     full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# documents = [Document(page_content=full_text)]  

print(docs)