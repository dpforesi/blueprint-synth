from collections import deque


class BlueprintCycleError(Exception):
    pass


class DAG:
    def __init__(self):
        self.nodes: list = []
        self.edges: list = []
        self._adj: dict = {}

    def add_node(self, name: str) -> None:
        if name not in self.nodes:
            self.nodes.append(name)
            self._adj[name] = []

    def add_edge(self, source: str, target: str) -> None:
        self.add_node(source)
        self.add_node(target)
        if (source, target) not in self.edges:
            self.edges.append((source, target))
            self._adj[source].append(target)

    def topological_sort(self) -> list:
        in_degree = {n: 0 for n in self.nodes}
        for src, tgt in self.edges:
            in_degree[tgt] += 1

        queue = deque(n for n in self.nodes if in_degree[n] == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self._adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            cycle_nodes = [n for n in self.nodes if n not in set(order)]
            raise BlueprintCycleError(f"Cycle detected involving: {cycle_nodes}")

        return order

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except BlueprintCycleError:
            return True
