from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


KNOWLEDGE_DIR = Path("knowledge")
FAISS_DIR = Path("faiss_index")


def load_documents():
    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):

        document = TextLoader(
            str(path),
            encoding="utf-8"
        ).load()[0]

        document.metadata.update(
            source=path.name,
            doc_type="advisory_knowledge",
        )

        documents.append(document)

    if not documents:
        raise ValueError(
            f"No knowledge documents found in {KNOWLEDGE_DIR}"
        )

    return documents


def build_vector_store():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        str(FAISS_DIR)
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"FAISS index saved to: {FAISS_DIR}")


if __name__ == "__main__":
    build_vector_store()