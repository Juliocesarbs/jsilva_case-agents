import pytest

from candidate_starter.harness import (
    compute_precision_at_k,
    compute_router_metrics,
    compute_savings,
)
from candidate_starter.retrieval import ToolRetriever
from candidate_starter.router import QueryRouter
from common.schemas import Tool


def test_router_requires_fit():
    router = QueryRouter()

    with pytest.raises(RuntimeError):
        router.predict("Quero saber meu saldo")


def test_retriever_requires_fit():
    retriever = ToolRetriever()

    with pytest.raises(RuntimeError):
        retriever.search("Quero saber meu saldo")


def test_retriever_respects_k():
    tools = [
        Tool(
            name="consultar_saldo",
            description="Consulta o saldo da conta.",
            category="conta",
        ),
        Tool(
            name="consultar_fatura",
            description="Consulta a fatura do cartão.",
            category="cartao",
        ),
        Tool(
            name="bloquear_cartao",
            description="Bloqueia um cartão.",
            category="cartao",
        ),
    ]

    retriever = ToolRetriever().fit(tools)

    result = retriever.search(
        "Quero consultar meu saldo",
        k=2,
    )

    assert len(result.matches) == 2
    assert result.matches[0].name == "consultar_saldo"


def test_router_metrics():
    y_true = [
        "FAST_PATH",
        "FAST_PATH",
        "AGENT",
        "AGENT",
    ]

    y_pred = [
        "FAST_PATH",
        "AGENT",
        "AGENT",
        "AGENT",
    ]

    labels = [
        "FAST_PATH",
        "AGENT",
    ]

    result = compute_router_metrics(
        y_true,
        y_pred,
        labels,
    )

    assert result["accuracy"] == pytest.approx(0.75)

    assert result["confusion_matrix"] == {
        "FAST_PATH": {
            "FAST_PATH": 1,
            "AGENT": 1,
        },
        "AGENT": {
            "FAST_PATH": 0,
            "AGENT": 2,
        },
    }


def test_precision_at_k_contract():
    hits = [1, 1, 0, 1]

    result = compute_precision_at_k(hits)

    assert result == pytest.approx(0.75)


def test_savings():
    result = compute_savings(
        smart_cost_usd=0.25,
        smart_latency_ms=400,
        baseline_cost_usd=1.00,
        baseline_latency_ms=1000,
    )

    assert result["cost_savings_pct"] == pytest.approx(75.0)
    assert result["latency_savings_pct"] == pytest.approx(60.0)