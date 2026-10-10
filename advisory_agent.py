
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


FAISS_DIR = Path("faiss_index")


def build_advisory_chain():
    # 1. Load the embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 2. Load the existing FAISS index
    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 3. Create a retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # 4. Define the advisory prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are an advisory assistant for a portfolio analysis
        application.

        Use the supplied financial knowledge and investor data
        to explain potential investment and rebalancing options.

        Do not invent missing portfolio values or financial facts.
        If information is insufficient, clearly state what is missing.
        Explain the rationale, relevant risks, and trade-offs.
        Do not present an illustrative allocation as a personalized
        recommendation without a defined allocation methodology.

        Investor profile:
        {investor_profile}

        Portfolio analysis:
        {portfolio_analysis}

        Risk assessment:
        {risk_assessment}

        Retrieved financial knowledge:
        {context}

        User question:
        {question}

        Provide a clear, structured advisory response.
        Include an appropriate suitability disclaimer.
        """
    )

    # 5. Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # 6. Connect retrieval, prompt, LLM, and output parser
    chain = (
        {
            "context": retriever | format_documents,
            "investor_profile": lambda x: x["investor_profile"],
            "portfolio_analysis": lambda x: x["portfolio_analysis"],
            "risk_assessment": lambda x: x["risk_assessment"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def format_documents(documents):
    return "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"{doc.page_content}"
        )
        for doc in documents
    )


def run_advisory(
    investor_profile: dict[str, Any],
    portfolio_analysis: dict[str, Any],
    risk_assessment: dict[str, Any],
    question: str,
):
    chain = build_advisory_chain()

    return chain.invoke(
        {
            "investor_profile": investor_profile,
            "portfolio_analysis": portfolio_analysis,
            "risk_assessment": risk_assessment,
            "question": question,
        }
    )


if __name__ == "__main__":
    # Illustrative test data, not a real investor assessment.
    result = run_advisory(
        investor_profile={
            "risk_tolerance": "moderate",
            "investment_horizon": "5 years",
            "investment_goal": "wealth creation",
        },
        portfolio_analysis={
            "concentration_risk": "high",
            "holdings": "Not provided in this test",
        },
        risk_assessment={
            "investor_risk_profile": "moderate",
            "portfolio_risk_profile": "high",
            "suitability": "mismatch",
            "key_risk_factors": ["High concentration risk"],
            "explanation": "Illustrative test assessment",
            "confidence": 0.91,
        },
        question=(
            "Explain what factors should be considered when "
            "evaluating rebalancing options for this investor."
        ),
    )

    print("\nADVISORY RESPONSE:\n")
    print(result)
