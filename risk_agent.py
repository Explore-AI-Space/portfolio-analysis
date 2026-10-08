import os
from typing import TypedDict, Literal, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# 2. Risk Agent State
# ============================================================

class RiskState(TypedDict, total=False):
    investor_profile: dict[str, Any]
    portfolio_analysis: dict[str, Any]
    risk_assessment: dict[str, Any]


# ============================================================
# 3. Structured output expected from GPT
# ============================================================

class RiskAssessment(BaseModel):
    investor_risk_profile: Literal[
        "conservative",
        "moderate",
        "aggressive"
    ]

    portfolio_risk_profile: Literal[
        "low",
        "moderate",
        "high"
    ]

    suitability: Literal[
        "match",
        "mismatch"
    ]

    key_risk_factors: list[str]

    explanation: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# 4. Create LLM
# ============================================================

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

structured_llm = llm.with_structured_output(
    RiskAssessment
)


# ============================================================
# 5. System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are the Risk Assessment Agent in an Agentic AI
investment portfolio system.

Your responsibilities are:

1. Assess the investor's risk profile.
2. Assess the risk characteristics of the current portfolio.
3. Determine whether the portfolio is suitable for the
   investor's stated risk tolerance.
4. Identify the primary factors influencing the assessment.

The portfolio analytics are produced by a separate
Analysis Agent.

Important rules:

- Use only the information provided.
- Do not invent missing data.
- Do not provide specific buy/sell recommendations.
- Do not confuse portfolio performance with portfolio risk.
- Consider multiple portfolio characteristics together.
- Return the result using the required structured schema.
"""


# ============================================================
# 6. Risk Assessment Node
# ============================================================

def risk_assessment_node(state: RiskState) -> RiskState:
    """
    Sends investor profile and portfolio analysis to the LLM
    and stores the structured risk assessment in state.
    """

    investor_profile = state.get("investor_profile")
    portfolio_analysis = state.get("portfolio_analysis")

    if not investor_profile:
        raise ValueError(
            "Missing investor_profile in Risk Agent state."
        )

    if not portfolio_analysis:
        raise ValueError(
            "Missing portfolio_analysis in Risk Agent state."
        )

    prompt = f"""
Investor Profile:
{investor_profile}

Portfolio Analysis:
{portfolio_analysis}

Assess:

1. The investor's risk profile.
2. The portfolio's overall risk profile.
3. Whether the portfolio is suitable for the investor.
4. The key factors driving the assessment.

Return the result using the required structured output.
"""

    result: RiskAssessment = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", prompt)
        ]
    )

    return {
        "risk_assessment": result.model_dump()
    }


# ============================================================
# 7. Build LangGraph
# ============================================================

def build_risk_agent():
    """
    Creates and compiles the LangGraph workflow.
    """

    builder = StateGraph(RiskState)

    # Add node
    builder.add_node(
        "risk_assessment",
        risk_assessment_node
    )

    # Workflow
    builder.add_edge(
        START,
        "risk_assessment"
    )

    builder.add_edge(
        "risk_assessment",
        END
    )

    return builder.compile()


# ============================================================
# 8. Mock Investor Profile
# ============================================================

investor_profile = {
    "risk_appetite": "moderate",
    "investment_horizon_years": 10,
    "loss_tolerance_percent": 15,
    "financial_goal": "wealth_creation",
    "investment_experience": "moderate"
}


# ============================================================
# 9. Mock Analysis Agent Output
# ============================================================

portfolio_analysis = {
    "portfolio_return_percent": 14.2,
    "volatility_percent": 21.4,
    "sharpe_ratio": 0.72,
    "equity_exposure_percent": 80.0,
    "sector_concentration_percent": 38.1,
    "largest_holding_weight_percent": 38.1
}


# ============================================================
# 10. Run Risk Agent
# ============================================================

def main():
    risk_agent = build_risk_agent()

    input_state: RiskState = {
        "investor_profile": investor_profile,
        "portfolio_analysis": portfolio_analysis
    }

    try:
        result = risk_agent.invoke(input_state)

        print("\n" + "=" * 60)
        print("RISK AGENT RESULT")
        print("=" * 60)

        print(result["risk_assessment"])

    except Exception as exc:
        print("\nRisk Agent execution failed:")
        print(str(exc))


# ============================================================
# 11. Entry Point
# ============================================================

if __name__ == "__main__":
    main()