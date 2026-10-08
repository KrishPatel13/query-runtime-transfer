from src.features import FEATURE_NAMES, extract_features

MOCK = [{"Plan": {
    "Node Type":"Aggregate","Startup Cost":10.0,"Total Cost":100.0,"Plan Rows":5,"Plan Width":16,
    "Plans":[{"Node Type":"Hash Join","Plan Rows":100,"Plan Width":20,
              "Plans":[{"Node Type":"Seq Scan","Plan Rows":1000,"Plan Width":8},
                       {"Node Type":"Hash","Plan Rows":50,"Plan Width":8,
                        "Plans":[{"Node Type":"Index Scan","Plan Rows":50,"Plan Width":8}]}]}]
}}]

PARALLEL = [{"Plan": {
    "Node Type":"Gather Merge","Startup Cost":1.0,"Total Cost":50.0,"Plan Rows":6,"Plan Width":8,"Workers Planned":2,
    "Plans":[{"Node Type":"Sort","Parent Relationship":"Outer","Plan Rows":3,"Plan Width":8,
              "Plans":[{"Node Type":"Seq Scan","Parent Relationship":"Outer","Plan Rows":300,"Plan Width":8},
                       {"Node Type":"Aggregate","Parent Relationship":"SubPlan","Plan Rows":1,"Plan Width":8}]}]
}}]


def test_basic_counts():
    f = extract_features(MOCK)
    assert set(f) == set(FEATURE_NAMES)
    assert f["total_cost"] == 100.0
    assert f["node_count"] == 5.0
    assert f["max_depth"] == 3.0
    assert f["hash_join_count"] == 1.0
    assert f["seq_scan_count"] == 1.0
    assert f["index_scan_count"] == 1.0
    assert f["aggregate_count"] == 1.0
    assert f["workers_planned"] == 0.0
    assert f["subplan_count"] == 0.0
    assert len(FEATURE_NAMES) == 19


def test_parallel_and_subplans():
    f = extract_features(PARALLEL)
    assert f["workers_planned"] == 2.0
    assert f["subplan_count"] == 1.0
    assert f["sort_count"] == 1.0


def test_cte_scan_counts_as_scan():
    plan = [{"Plan": {"Node Type": "Hash Join", "Total Cost": 5.0, "Plans": [
        {"Node Type": "CTE Scan", "Plan Rows": 10},
        {"Node Type": "Seq Scan", "Plan Rows": 10}]}}]
    assert extract_features(plan)["scan_count"] == 2.0


if __name__ == "__main__":
    test_basic_counts()
    test_parallel_and_subplans()
    print("feature extraction tests passed")
