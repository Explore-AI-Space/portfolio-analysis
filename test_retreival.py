from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


FAISS_DIR = Path("faiss_index")


def test_retrieval():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    question = (
        "What should be considered when recommending "
        "changes for a moderate-risk investor?"
    )

    documents = retriever.invoke(question)

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED DOCUMENTS:")

    for i, document in enumerate(documents, start=1):
        print(f"\n--- Document {i} ---")
        print("Source:", document.metadata.get("source"))
        print("Content:", document.page_content)


if __name__ == "__main__":
    test_retrieval()