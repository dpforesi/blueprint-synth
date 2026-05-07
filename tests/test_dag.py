import pytest
from blueprint.core.dag import DAG, BlueprintCycleError


class TestDAGNodes:
    def test_add_node(self):
        dag = DAG()
        dag.add_node("sqft")
        assert "sqft" in dag.nodes

    def test_duplicate_node_is_idempotent(self):
        dag = DAG()
        dag.add_node("sqft")
        dag.add_node("sqft")
        assert dag.nodes.count("sqft") == 1 or len([n for n in dag.nodes if n == "sqft"]) == 1


class TestDAGEdges:
    def test_add_edge(self):
        dag = DAG()
        dag.add_node("sqft")
        dag.add_node("price")
        dag.add_edge("sqft", "price")
        assert ("sqft", "price") in dag.edges

    def test_add_edge_implicitly_adds_nodes(self):
        dag = DAG()
        dag.add_edge("sqft", "price")
        assert "sqft" in dag.nodes
        assert "price" in dag.nodes


class TestTopologicalSort:
    def test_independent_nodes_all_present(self):
        dag = DAG()
        for name in ["a", "b", "c"]:
            dag.add_node(name)
        order = dag.topological_sort()
        assert set(order) == {"a", "b", "c"}

    def test_dependency_comes_before_dependent(self):
        dag = DAG()
        dag.add_edge("sqft", "price")
        order = dag.topological_sort()
        assert order.index("sqft") < order.index("price")

    def test_chain_resolved_in_order(self):
        dag = DAG()
        dag.add_edge("sqft", "price")
        dag.add_edge("price", "price_per_sqft")
        order = dag.topological_sort()
        assert order.index("sqft") < order.index("price")
        assert order.index("price") < order.index("price_per_sqft")

    def test_real_estate_dag(self):
        dag = DAG()
        dag.add_edge("neighborhood_tier", "has_pool")
        dag.add_edge("sqft", "price")
        dag.add_edge("bedrooms", "price")
        dag.add_edge("has_pool", "price")
        dag.add_edge("crime_rate", "price")
        dag.add_edge("price", "price_per_sqft")
        dag.add_edge("sqft", "price_per_sqft")
        order = dag.topological_sort()
        assert order.index("neighborhood_tier") < order.index("has_pool")
        assert order.index("has_pool") < order.index("price")
        assert order.index("price") < order.index("price_per_sqft")

    def test_no_edges_returns_all_nodes(self):
        dag = DAG()
        for n in ["a", "b", "c", "d"]:
            dag.add_node(n)
        order = dag.topological_sort()
        assert len(order) == 4


class TestCycleDetection:
    def test_self_loop_raises(self):
        dag = DAG()
        dag.add_edge("price", "price")
        with pytest.raises(BlueprintCycleError):
            dag.topological_sort()

    def test_two_node_cycle_raises(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(BlueprintCycleError):
            dag.topological_sort()

    def test_three_node_cycle_raises(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")
        with pytest.raises(BlueprintCycleError):
            dag.topological_sort()

    def test_cycle_error_includes_path(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        with pytest.raises(BlueprintCycleError, match="a|b"):
            dag.topological_sort()

    def test_has_cycle_true(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")
        assert dag.has_cycle() is True

    def test_has_cycle_false(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        assert dag.has_cycle() is False


class TestDAGEdgeCases:
    def test_empty_dag_sort_returns_empty(self):
        dag = DAG()
        assert dag.topological_sort() == []

    def test_empty_dag_no_cycle(self):
        dag = DAG()
        assert dag.has_cycle() is False

    def test_diamond_dependency(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "d")
        dag.add_edge("c", "d")
        order = dag.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_duplicate_edge_not_double_counted(self):
        dag = DAG()
        dag.add_edge("a", "b")
        dag.add_edge("a", "b")
        assert dag.edges.count(("a", "b")) == 1
