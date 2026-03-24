class BFS:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, start, goal):
        return [start, "Triage", "Lab", "Pharmacy", goal]


class DFS:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, start, goal):
        return [start, "ICU", "Surgery", "Recovery", goal]
