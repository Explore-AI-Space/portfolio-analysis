
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
            "Whether the advisory acknowledges and does not "
            "contradict the supplied risk assessment."
        )
    )

    suitability_check: bool = Field(
        description=(
            "Whether the advisory appropriately considers the "
            "investor profile without unsupported suitability claims."
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
    
    # Deterministic disclaimer-presence check
    advisory = state["advisory"]
    disclaimer = advisory.get("disclaimer", "")
    
    disclaimer_present = (
        isinstance(disclaimer, str)
        and bool(disclaimer.strip())
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
    1. RISK CONSISTENCY

    Return True if the advisory acknowledges and does not contradict
    the supplied risk assessment.

    A recommendation to review concentration or diversification is
    consistent with a high portfolio risk assessment.

    Return False if the advisory contradicts or materially ignores
    a risk in a way that could make the advice misleading.

    2. SUITABILITY CHECK

    Return True if the advisory appropriately considers the supplied
    investor profile and avoids unsupported suitability claims.

    An exact allocation is not required to pass this check.

    Return False if the advisory recommends an approach that
    conflicts with the investor profile or makes unsupported
    personalized suitability claims.

    If the information is insufficient to assess suitability,
    mention the limitation in issues. Use needs_review as the
    overall status when appropriate.

    3. FACTUAL GROUNDING

    Check whether material financial claims are supported by the
    supplied inputs.

    Flag invented portfolio values, unsupported financial claims,
    and guaranteed-return claims.

    Do not assume a claim is correct merely because another agent
    states it.

    4. DISCLAIMER

    The deterministic disclaimer-presence check returned:
    {disclaimer_present}

    Set disclaimer_present to this exact Boolean value.

    A non-empty disclaimer is not necessarily an adequate disclaimer.
    If the wording is inadequate, explain the limitation in issues
    and choose the overall status accordingly.

    5. OVERALL STATUS

    - pass: no material issues identified.
    - needs_review: uncertainty or correctable issues remain.
    - fail: material problems make the advisory unsuitable for release.

    Do not claim that regulatory compliance has been established.
    Do not invent missing information.
    Explain the reasons for any failed checks.
    Provide specific required corrections when necessary.

    Return the result using the required QAResult schema.
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
