
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class QAResult(BaseModel):
    status: Literal["pass", "needs_review", "fail"] = Field(
        description=(
            "Overall quality status. Pass when no material issues "
            "are found, needs_review when clarification or human "
            "review is required, and fail when material issues "
            "make the advisory unsuitable for release."
        )
    )

    risk_consistency: bool = Field(
        description=(
            "Whether the advisory is consistent with the supplied "
            "risk assessment."
        )
    )

    suitability_check: bool = Field(
        description=(
            "Whether the advisory considers the investor's risk "
            "profile and available financial goals."
        )
    )

    factual_grounding: bool = Field(
        description=(
            "Whether material factual claims are supported by "
            "the supplied inputs and retrieved knowledge."
        )
    )

    disclaimer_present: bool = Field(
        description="Whether an appropriate disclaimer is present."
    )

    issues: list[str] = Field(
        description="Specific problems found during the review."
    )

    required_corrections: list[str] = Field(
        description="Changes required before the advisory is released."
    )

    explanation: str = Field(
        description="Brief explanation of the QA decision."
    )


class QAState(TypedDict, total=False):
    investor_profile: dict[str, Any]
    portfolio_analysis: dict[str, Any]
    risk_assessment: dict[str, Any]
    advisory: dict[str, Any]
    qa_review: dict[str, Any]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

structured_llm = llm.with_structured_output(QAResult)


def qa_node(state: QAState) -> dict:
    required_fields = [
        "investor_profile",
        "portfolio_analysis",
        "risk_assessment",
        "advisory",
    ]

    missing = [
        field for field in required_fields
        if field not in state or not state[field]
    ]

    if missing:
        raise ValueError(
            f"QA Agent is missing required inputs: {missing}"
        )

    prompt = f"""
    You are the QA Agent for a portfolio advisory application.

    Review the advisory against the supplied investor profile,
    portfolio analysis, risk assessment, and advisory knowledge
    available in the inputs.

    Investor Profile:
    {state['investor_profile']}

    Portfolio Analysis:
    {state['portfolio_analysis']}

    Risk Assessment:
    {state['risk_assessment']}

    Advisory:
    {state['advisory']}

    Review instructions:
    1. Check whether the advisory contradicts the risk assessment.
    2. Check whether suitability considers the investor profile.
    3. Identify material financial claims unsupported by the inputs.
    4. Flag invented portfolio values, unjustified allocations,
       and guaranteed-return claims.
    5. Check whether an appropriate disclaimer is present.
    6. Do not assume a claim is correct merely because the advisory
       or risk assessment states it.
    7. Do not claim regulatory compliance has been established.
    8. If information is insufficient to verify a material claim,
       flag it for review rather than assuming it is true.
    9. Set status to:
       - pass: no material issues identified;
       - needs_review: uncertainty or correctable issues remain;
       - fail: material problems make the advisory unsuitable
         for release.
    10. Explain specific issues and required corrections.
        Do not rewrite the advisory yourself.
    """

    result = structured_llm.invoke(prompt)

    return {
        "qa_review": result.model_dump()
    }


if __name__ == "__main__":
    test_state = {
        "investor_profile": {
            "risk_tolerance": "moderate",
            "investment_horizon": "5 years",
            "investment_goal": "wealth creation",
        },
        "portfolio_analysis": {
            "concentration_risk": "high",
            "holdings": "Not provided",
        },
        "risk_assessment": {
            "investor_risk_profile": "moderate",
            "portfolio_risk_profile": "high",
            "suitability": "mismatch",
            "key_risk_factors": ["High concentration risk"],
        },
        "advisory": {
            "recommendation": (
                "Evaluate concentration and diversification."
            ),
            "rationale": [
                "Concentration may increase portfolio risk."
            ],
            "key_risks": ["Concentration risk"],
            "rebalancing_actions": [
                "Review actual holdings before deciding."
            ],
            "disclaimer": (
                "This is informational and not a guarantee "
                "of investment results."
            ),
        },
    }

    result = qa_node(test_state)

    print("\nQA REVIEW:\n")
    print(result["qa_review"])
