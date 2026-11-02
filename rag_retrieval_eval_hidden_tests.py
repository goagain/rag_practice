# -*- coding: utf-8 -*-
"""Hidden checks for rag_retrieval_eval_practice.ipynb.

The practice notebook only calls run(). Keep this file closed until the
blanks are filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cosine(q, D):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    D = np.asarray(D, dtype=np.float64)
    denom = np.linalg.norm(D, axis=1) * np.linalg.norm(q)
    sims = np.zeros(D.shape[0], dtype=np.float64)
    np.divide(D @ q, denom, out=sims, where=denom > 0)
    return sims


def _as_str_list(values):
    return [str(value) for value in list(values)]


def _is_id_sequence(values):
    """Accept a Python list or a one-dimensional array of document ids."""
    if isinstance(values, (str, bytes)):
        return False
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            return False
        items = values.tolist()
    elif isinstance(values, (list, tuple)):
        items = list(values)
    else:
        return False
    return all(isinstance(item, str) for item in items)


# Objective points. Performance (15) and style (15) are not decided here.
_LOGIC_POINTS = {
    "build_doc_embeddings": 10,
    "cosine_similarity": 18,
    "top_k_doc_ids": 14,
    "recall_at_k": 5,
    "precision_at_k": 5,
    "hit_at_k": 4,
    "mrr_at_k": 6,
    "evaluate": 8,
}
_LOGIC_CHECKS = {
    "build_doc_embeddings": [
        "build_texts_in_row_order",
        "build_doc_id_order",
        "build_shape",
        "build_rows_match_texts",
    ],
    "cosine_similarity": [
        "cosine_shape",
        "cosine_values",
        "cosine_q_shapes_agree",
        "cosine_opposite",
        "cosine_orthogonal",
        "cosine_zero_row",
        "cosine_zero_query",
    ],
    "top_k_doc_ids": [
        "topk_id_sequence",
        "topk_order",
        "topk_k_gt_n",
        "topk_large_order",
    ],
    "recall_at_k": [
        "recall_rank11_excluded",
        "recall_empty_relevant",
        "recall_duplicate_relevant",
    ],
    "precision_at_k": [
        "precision_denominator_is_k",
        "precision_duplicate_retrieved",
    ],
    "hit_at_k": ["hit_rank11_excluded"],
    "mrr_at_k": ["mrr_second_place", "mrr_after_k"],
    "evaluate": [
        "evaluate_calls_each_query",
        "evaluate_keys",
        "evaluate_Recall@2",
        "evaluate_Precision@2",
        "evaluate_Hit@2",
        "evaluate_MRR@2",
    ],
}


class _Runner:
    def __init__(self, ns):
        self.ns = ns
        self.failures = []
        self.passed = 0
        self.passed_names = set()

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            self.passed_names.add(name)
            return
        message = name if not detail else f"{name}: {detail}"
        self.failures.append(message)

    def need(self, name):
        fn = self.ns.get(name)
        if not callable(fn):
            self.failures.append(f"{name}: function is not defined")
            return None
        return fn


def _test_cosine(runner):
    cosine = runner.need("cosine_similarity")
    if cosine is None:
        return
    rng = np.random.default_rng(11)
    D = rng.normal(size=(7, 5))
    q_row = rng.normal(size=(1, 5))
    q_flat = q_row.reshape(-1)
    try:
        got_row = np.asarray(cosine(q_row, D), dtype=np.float64)
        got_flat = np.asarray(cosine(q_flat, D), dtype=np.float64)
        expected = _cosine(q_flat, D)
    except Exception as exc:
        runner.failures.append(f"cosine_random: {type(exc).__name__}: {exc}")
        return
    runner.check("cosine_shape", got_row.shape == (7,), f"shape is {getattr(got_row, 'shape', None)}")
    runner.check("cosine_values", np.allclose(got_row, expected, rtol=1e-6, atol=1e-8))
    runner.check("cosine_q_shapes_agree", np.allclose(got_row, got_flat, rtol=1e-6, atol=1e-8))

    opposite = np.array([[-2.0, 0.0], [0.0, 3.0], [0.0, 0.0]], dtype=float)
    try:
        sims = np.asarray(cosine(np.array([2.0, 0.0]), opposite), dtype=np.float64)
    except Exception as exc:
        runner.failures.append(f"cosine_signs: {type(exc).__name__}: {exc}")
        return
    runner.check("cosine_opposite", sims.shape == (3,) and np.isclose(sims[0], -1.0), np.round(sims, 5))
    runner.check("cosine_orthogonal", np.isclose(sims[1], 0.0), np.round(sims, 5))
    runner.check(
        "cosine_zero_row",
        np.isfinite(sims).all() and np.isclose(sims[2], 0.0),
        np.round(sims, 5),
    )

    try:
        zero_q = np.asarray(cosine(np.zeros(2), opposite), dtype=np.float64)
    except Exception as exc:
        runner.failures.append(f"cosine_zero_query: {type(exc).__name__}: {exc}")
        return
    runner.check(
        "cosine_zero_query",
        zero_q.shape == (3,) and np.isfinite(zero_q).all() and np.allclose(zero_q, 0.0),
        np.round(zero_q, 5),
    )


def _ranked_ids(q, D, ids):
    order = np.argsort(-_cosine(q, D), kind="mergesort")
    return [ids[i] for i in order]


def _test_topk(runner):
    topk = runner.need("top_k_doc_ids")
    if topk is None:
        return
    thetas = np.linspace(-2.5, 2.2, 9)
    D = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    q = np.array([1.0, 0.0])
    ids = [f"d{i}" for i in range(len(thetas))]
    try:
        got = topk(q, D, ids, k=4)
    except Exception as exc:
        runner.failures.append(f"topk_order: {type(exc).__name__}: {exc}")
        return
    expected = _ranked_ids(q, D, ids)[:4]
    runner.check("topk_id_sequence", _is_id_sequence(got), type(got))
    runner.check("topk_order", _as_str_list(got) == expected, f"got {_as_str_list(got)}, expected {expected}")

    try:
        got_all = topk(q, D, list(ids), k=20)
    except Exception as exc:
        runner.failures.append(f"topk_k_gt_n: {type(exc).__name__}: {exc}")
        return
    expected_all = _ranked_ids(q, D, ids)
    runner.check(
        "topk_k_gt_n",
        _is_id_sequence(got_all) and _as_str_list(got_all) == expected_all,
        f"got {_as_str_list(got_all)}, expected {expected_all}",
    )

    rng = np.random.default_rng(19)
    n, k = 1800, 250
    angles = rng.uniform(-np.pi, np.pi, size=n)
    D_large = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    labels = np.array([f"p{i:04d}" for i in range(n)])
    try:
        got_large = topk(q, D_large, labels, k=k)
    except Exception as exc:
        runner.failures.append(f"topk_large_order: {type(exc).__name__}: {exc}")
        return
    expected_large = _ranked_ids(q, D_large, labels.tolist())[:k]
    head = 6
    runner.check(
        "topk_large_order",
        _as_str_list(got_large) == expected_large,
        f"first {head} got {_as_str_list(got_large)[:head]}, expected {expected_large[:head]}",
    )


def _test_metrics(runner):
    recall = runner.need("recall_at_k")
    precision = runner.need("precision_at_k")
    hit = runner.need("hit_at_k")
    mrr = runner.need("mrr_at_k")
    if not all([recall, precision, hit, mrr]):
        return
    ranked = [f"x{i}" for i in range(12)]
    relevant = ["x10", "x3"]
    try:
        runner.check("recall_rank11_excluded", np.isclose(recall(ranked, relevant, 10), 0.5))
        runner.check("hit_rank11_excluded", np.isclose(hit(ranked, ["x11"], 10), 0.0))
        runner.check("mrr_second_place", np.isclose(mrr(ranked, relevant, 10), 1 / 4))
        runner.check("mrr_after_k", np.isclose(mrr(ranked, ["x10"], 10), 0.0))
        runner.check("precision_denominator_is_k", np.isclose(precision(["x3"], relevant, 10), 0.1))
        runner.check("recall_empty_relevant", np.isclose(recall(["x0"], [], 10), 0.0))
        runner.check("recall_duplicate_relevant", np.isclose(recall(["a", "a", "b"], ["a", "b", "a"], 10), 1.0))
        runner.check("precision_duplicate_retrieved", np.isclose(precision(["a", "a"], ["a"], 4), 0.25))
    except Exception as exc:
        runner.failures.append(f"metrics: {type(exc).__name__}: {exc}")


def _test_evaluate(runner):
    evaluate = runner.need("evaluate")
    if evaluate is None:
        return
    calls = {"n": 0}
    original = runner.ns.get("embed_query")

    def embed_query(text):
        calls["n"] += 1
        return np.array([1.0, 0.0])

    runner.ns["embed_query"] = embed_query
    D = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    doc_ids = ["d0", "d1", "d2"]
    queries = pd.DataFrame(
        {
            "query_id": ["qa", "qb", "qc"],
            "query": ["north", "south", "both"],
            "relevant_doc_ids": [["d0"], ["d2"], ["d0", "d1"]],
        }
    )
    try:
        metrics = evaluate(queries, D, doc_ids, k=2)
    except Exception as exc:
        runner.failures.append(f"evaluate: {type(exc).__name__}: {exc}")
        return
    finally:
        if original is not None:
            runner.ns["embed_query"] = original

    expected = {
        "Recall@2": (1.0 + 0.0 + 1.0) / 3.0,
        "Precision@2": (0.5 + 0.0 + 1.0) / 3.0,
        "Hit@2": (1.0 + 0.0 + 1.0) / 3.0,
        "MRR@2": (1.0 + 0.0 + 1.0) / 3.0,
    }
    runner.check("evaluate_calls_each_query", calls["n"] == 3, f"embed_query was called {calls['n']} times")
    if not isinstance(metrics, dict):
        runner.failures.append("evaluate_keys: return value is not a dict")
        return
    missing = [key for key in expected if key not in metrics]
    runner.check("evaluate_keys", not missing, f"missing {missing}")
    if not missing:
        for key, value in expected.items():
            runner.check(f"evaluate_{key}", np.isclose(metrics[key], value), f"got {metrics[key]}")


def _test_build(runner):
    build = runner.need("build_doc_embeddings")
    if build is None:
        return
    seen = {"texts": []}
    original = runner.ns.get("embed_texts")

    def embed_texts(texts):
        texts = [str(text) for text in texts]
        seen["texts"].extend(texts)
        rows = [[float(len(text)), float(sum(ord(ch) for ch in text) % 1000)] for text in texts]
        return np.asarray(rows, dtype=np.float64)

    runner.ns["embed_texts"] = embed_texts
    docs = pd.DataFrame(
        {
            "doc_id": ["d2", "d0", "d1"],
            "title": ["t2", "t0", "t1"],
            "topic": ["z", "a", "m"],
            "text": ["gamma", "alpha", "beta"],
        }
    )
    docs = docs.iloc[[2, 0, 1]]
    try:
        D, doc_ids = build(docs)
    except Exception as exc:
        runner.failures.append(f"build_doc_embeddings: {type(exc).__name__}: {exc}")
        return
    finally:
        if original is not None:
            runner.ns["embed_texts"] = original

    D = np.asarray(D, dtype=np.float64)
    got_ids = _as_str_list(doc_ids)
    expected_ids = docs["doc_id"].astype(str).tolist()
    expected_texts = docs["text"].astype(str).tolist()
    runner.check("build_texts_in_row_order", seen["texts"] == expected_texts, seen["texts"])
    runner.check("build_doc_id_order", got_ids == expected_ids, got_ids)
    runner.check("build_shape", D.shape == (3, 2), getattr(D, "shape", None))
    if D.shape == (3, 2) and seen["texts"] == expected_texts:
        expected_D = embed_texts(expected_texts)
        runner.ns["embed_texts"] = original
        runner.check("build_rows_match_texts", np.allclose(D, expected_D))


def _format_points(value):
    rounded = round(value, 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def _print_objective_score(runner):
    total = 0.0
    print("Objective score (logic)")
    for name, points in _LOGIC_POINTS.items():
        checks = _LOGIC_CHECKS[name]
        got = sum(1 for check in checks if check in runner.passed_names)
        score = float(points) if got == len(checks) else points * got / len(checks)
        total += score
        print(f"  {name}: {_format_points(score)}/{points}")
    print(f"Logic: {_format_points(total)}/70")
    print("Performance (15) and style (15) are not included.")


def run(namespace):
    """Run the hidden checks against functions defined in ``namespace``."""
    runner = _Runner(namespace)
    _test_cosine(runner)
    _test_topk(runner)
    _test_metrics(runner)
    _test_evaluate(runner)
    _test_build(runner)
    total = runner.passed + len(runner.failures)
    if runner.failures:
        print(f"Hidden tests passed {runner.passed}/{total}")
        for message in runner.failures:
            print(f"- {message}")
    else:
        print(f"Hidden tests passed {runner.passed}/{total}")
    _print_objective_score(runner)
    return not runner.failures
