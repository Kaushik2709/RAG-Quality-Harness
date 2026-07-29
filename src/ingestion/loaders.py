import os
from typing import List
from langchain_core.documents import Document

def load_document(file_path: str) -> List[Document]:
    """
    Loads text, HTML, or PDF documents into LangChain Document objects without heavy dependencies.
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    try:
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages = []
                for idx, page in enumerate(reader.pages):
                    content = page.extract_text() or ""
                    if content.strip():
                        pages.append(Document(
                            page_content=content,
                            metadata={"source": filename, "file_name": filename, "doc_id": filename, "page": idx + 1}
                        ))
                return pages or [Document(page_content="", metadata={"source": filename, "doc_id": filename})]
            except Exception:
                from langchain_community.document_loaders import PyPDFLoader
                return PyPDFLoader(file_path).load()

        elif ext in [".html", ".htm"]:
            try:
                from bs4 import BeautifulSoup
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    text = soup.get_text(separator="\n")
                return [Document(page_content=text, metadata={"source": filename, "file_name": filename, "doc_id": filename})]
            except Exception:
                from langchain_community.document_loaders import BSHTMLLoader
                return BSHTMLLoader(file_path).load()

        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": filename, "file_name": filename, "doc_id": filename})]

    except Exception as e:
        print(f"[Loader Warning] Error loading {file_path}: {e}")
        return [Document(page_content="", metadata={"source": filename, "doc_id": filename, "error": str(e)})]

