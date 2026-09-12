"""Pilar 3 — Evaluation Harness (Evals & Benchmarking).

A orquestração do pipeline (rodar o router, a seleção de tools e os mocks de LLM/tool) já
está feita em `run_harness`. O que falta implementar são as MÉTRICAS do relatório final:

  1. Acurácia do Router (+ matriz de confusão).
  2. Precision@K do Retriever de Tools (a tool certa estava no Top-K?).
  3. Economia de custo e latência do pipeline "inteligente" (router + seleção de tools)
     comparado ao baseline de mandar tudo para o LLM mais caro.

Preencha as funções marcadas com TODO. Respeite o formato de retorno pedido em cada
docstring, pois `run_harness` e `print_report` dependem dessas chaves.
"""

import time
from typing import Dict, List

from common.interfaces import BaseRouter, BaseToolRetriever
from common.mock_llm import (
    COST_RETRIEVAL_USD,
    COST_ROUTER_USD,
    fast_path_answer,
    mock_tool_execution,
    simulate_agent_llm_call,
    simulate_baseline_llm_call,
)
from common.schemas import Tool


def compute_router_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict:
    """Calcula a acurácia e a matriz de confusão do router."""
    total = len(y_true)

    accuracy = (
        sum(true == pred for true, pred in zip(y_true, y_pred)) / total
        if total
        else 0.0
    )

    confusion_matrix = {
        true_label: {
            pred_label: sum(
                true == true_label and pred == pred_label
                for true, pred in zip(y_true, y_pred)
            )
            for pred_label in labels
        }
        for true_label in labels
    }

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
    }


def compute_precision_at_k(hits: List[int]) -> float:
    """Calcula a taxa de queries em que a tool esperada apareceu no Top-K."""
    if not hits:
        return 0.0

    return sum(hits) / len(hits)


def compute_savings(
    smart_cost_usd: float,
    smart_latency_ms: float,
    baseline_cost_usd: float,
    baseline_latency_ms: float,
) -> Dict:
    """Calcula a economia percentual de custo e latência."""
    cost_savings_pct = (
        (1 - smart_cost_usd / baseline_cost_usd) * 100
        if baseline_cost_usd
        else 0.0
    )

    latency_savings_pct = (
        (1 - smart_latency_ms / baseline_latency_ms) * 100
        if baseline_latency_ms
        else 0.0
    )

    return {
        "cost_savings_pct": cost_savings_pct,
        "latency_savings_pct": latency_savings_pct,
    }


def run_harness(
    router: BaseRouter,
    retriever: BaseToolRetriever,
    tools: List[Tool],
    eval_dataset: List[dict],
    k: int = 2,
) -> dict:
    labels = ["FAST_PATH", "AGENT"]

    y_true: List[str] = []
    y_pred: List[str] = []
    precision_hits: List[int] = []
    smart_cost_total = 0.0
    smart_latency_ms_total = 0.0
    baseline_cost_total = 0.0
    baseline_latency_ms_total = 0.0

    rows = []

    for item in eval_dataset:
        query = item["query"]
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")

        route_result = router.predict(query)
        y_true.append(expected_route)
        y_pred.append(route_result.route)
        smart_cost = COST_ROUTER_USD
        smart_latency_ms = route_result.latency_ms

        row = {
            "query": query,
            "expected_route": expected_route,
            "predicted_route": route_result.route,
        }

        if route_result.route == "FAST_PATH":
            fast_path_answer(query)
        else:
            retrieval_result = retriever.search(query, k=k)
            smart_cost += COST_RETRIEVAL_USD
            smart_latency_ms += retrieval_result.latency_ms
            top_k_names = [m.name for m in retrieval_result.matches]

            if expected_tool:
                precision_hits.append(int(expected_tool in top_k_names))

            row["retrieved_tools"] = top_k_names
            row["expected_tool"] = expected_tool

            if top_k_names:
                mock_tool_execution(top_k_names[0], query)
                llm_result = simulate_agent_llm_call(query, top_k_names[0])
                smart_cost += llm_result["cost_usd"]

        smart_cost_total += smart_cost
        smart_latency_ms_total += smart_latency_ms

        baseline_start = time.perf_counter()
        baseline_result = simulate_baseline_llm_call(query)
        baseline_latency_ms_total += (
            time.perf_counter() - baseline_start
        ) * 1000
        baseline_cost_total += baseline_result["cost_usd"]

        rows.append(row)

    router_metrics = compute_router_metrics(
        y_true,
        y_pred,
        labels,
    )

    precision_at_k = (
        compute_precision_at_k(precision_hits)
        if precision_hits
        else None
    )

    savings = compute_savings(
        smart_cost_total,
        smart_latency_ms_total,
        baseline_cost_total,
        baseline_latency_ms_total,
    )

    report = {
        "n_queries": len(eval_dataset),
        "router_accuracy": router_metrics["accuracy"],
        "confusion_matrix": router_metrics["confusion_matrix"],
        "precision_at_k": precision_at_k,
        "k": k,
        "smart_pipeline": {
            "total_cost_usd": smart_cost_total,
            "total_latency_ms": smart_latency_ms_total,
        },
        "baseline_always_llm": {
            "total_cost_usd": baseline_cost_total,
            "total_latency_ms": baseline_latency_ms_total,
        },
        **savings,
        "rows": rows,
    }

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("HARNESS DE AVALIAÇÃO - Router & Tool Retrieval")
    print("=" * 60)
    print(f"Queries avaliadas: {report['n_queries']}")
    print(f"Acurácia do Router: {report['router_accuracy']:.1%}")
    print(f"Matriz de confusão: {report['confusion_matrix']}")

    if report["precision_at_k"] is not None:
        print(
            f"Precision@{report['k']} do Retriever: "
            f"{report['precision_at_k']:.1%}"
        )

    print("-" * 60)
    print(
        f"Custo pipeline inteligente: "
        f"${report['smart_pipeline']['total_cost_usd']:.5f}"
    )
    print(
        f"Custo baseline (tudo pro LLM): "
        f"${report['baseline_always_llm']['total_cost_usd']:.5f}"
    )
    print(
        f"Economia de custo: "
        f"{report.get('cost_savings_pct', 0):.1f}%"
    )
    print(
        f"Latência pipeline inteligente: "
        f"{report['smart_pipeline']['total_latency_ms']:.1f} ms"
    )
    print(
        f"Latência baseline: "
        f"{report['baseline_always_llm']['total_latency_ms']:.1f} ms"
    )
    print(
        f"Economia de latência: "
        f"{report.get('latency_savings_pct', 0):.1f}%"
    )
    print("=" * 60)