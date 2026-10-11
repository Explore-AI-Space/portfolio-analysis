
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from risk_agent import build_risk_agent
from advisory_agent import advisory_node


class PortfolioAdvisorState(TypedDict, total=False):
    investor_profile: dict[str, Any]
    portfolio_analysis: dict[str, Any]
    risk_assessment: dict[str, Any]
    question: str
    advisory: dict[str, Any]


def build_orchestrator():
    # Create the main graph
    graph = StateGraph(PortfolioAdvisorState)

    # Register the existing Risk Agent graph
    graph.add_node("risk_agent", build_risk_agent())

    # Register the Advisory Agent as a node
    graph.add_node("advisory_agent", advisory_node)

    # Define execution order
    graph.add_edge(START, "risk_agent")
    graph.add_edge("risk_agent", "advisory_agent")
    graph.add_edge("advisory_agent", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_orchestrator()

    initial_state = {
        "investor_profile": {
            "risk_appetite": "moderate",
            "investment_horizon_years": 10,
            "loss_tolerance_percent": 15,
            "financial_goal": "wealth_creation",
            "investment_experience": "moderate",
        },
        "portfolio_analysis": {
            "portfolio_return_percent": 14.2,
            "volatility_percent": 21.4,
            "sharpe_ratio": 0.72,
            "equity_exposure_percent": 80.0,
            "sector_concentration_percent": 38.1,
            "largest_holding_weight_percent": 38.1,
        },
        "question": (
            "What factors should be considered when "
            "evaluating rebalancing options for this portfolio?"
        ),
    }

    result = app.invoke(initial_state)

    print("\nRISK ASSESSMENT:\n")
    print(result["risk_assessment"])

    print("\nADVISORY:\n")
    print(result["advisory"])
