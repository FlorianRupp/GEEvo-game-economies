import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Edge:
    def __init__(self, node, value, name=None, node_id=None):
        self.value = value
        self.node = node
        if name is not None:
            self.node.name = name
        self.node_id = node_id


class Node(ABC):
    ALLOWED_INPUT = []
    ALLOWED_OUTPUT = []
    EMPTY_INPUT = True
    EMPTY_OUTPUT = True
    MAX_INPUT = 0
    MAX_OUTPUT = 0
    COLOR = None

    def __init__(self, name=None, id=None):
        self.name = name
        self.input_edges = []
        self.output_edges = []
        self.id = id

    def check(self, node):
        if type(self) not in node.ALLOWED_INPUT or type(node) not in self.ALLOWED_OUTPUT:
            raise ValueError(f"Connections of {type(self)} and {type(node)} are not allowed.")
        node_edges = [e.node for e in node.input_edges]
        node_edges.extend([e.node for e in node.output_edges])
        if self in node_edges:
            raise ValueError(f"Nodes are already connected.")
        if node.MAX_INPUT == len(node.input_edges):
            raise ValueError(f"{type(node).__name__} has already the maximum of {node.MAX_INPUT} inputs.")
        if self.MAX_OUTPUT == len(self.output_edges):
            raise ValueError(f"{type(self).__name__} has already the maximum of {self.MAX_OUTPUT} outputs.")

    def check_connection(func):
        def wrapper(self, node, value, name=None, node_id=None):
            self.check(node)
            func(self, node, value, name, node_id)

        return wrapper

    @staticmethod
    def init_or_random(value, a=0, b=2, integer=True):
        if value is None:
            if integer is True:
                return random.randint(a, b)
            else:
                return random.random()
        else:
            return value

    @check_connection
    def connect(self, node, value, name=None, node_id=None):
        self.output_edges.append(Edge(node, value, name, node_id=node_id))
        node.input_edges.append(Edge(self, value, name))

    def disconnect(self, node):
        assert node in self.get_output_nodes(), "Nodes are not connected"
        self.output_edges.remove([e for e in self.output_edges if e.node is node][0])
        node.input_edges.remove([e for e in node.input_edges if e.node is self][0])

    def get_output_nodes(self):
        return [edge.node for edge in self.output_edges]

    def get_input_nodes(self):
        return [edge.node for edge in self.input_edges]

    def step(self, call_chain):
        pass

    def __str__(self):
        input_edges = [type(e.node).__name__ for e in self.input_edges]
        output_edges = [type(e.node).__name__ for e in self.output_edges]
        return f"{type(self).__name__}: [input: {input_edges}, output: {output_edges}]"

    def update_edge_value(self, node, value):
        [e for e in self.output_edges if e.node == node][0].value = value
        [e for e in node.input_edges if e.node == self][0].value = value

    def get_state(self):
        score = 0
        if len(self.input_edges) == 0 and self.EMPTY_INPUT is False:
            score -= 1
        if len(self.output_edges) == 0 and self.EMPTY_OUTPUT is False:
            score -= 1
        return score


class Source(Node):
    EMPTY_INPUT = True
    EMPTY_OUTPUT = False
    MAX_INPUT = 0
    MAX_OUTPUT = 3
    COLOR = "limegreen"

    def step(self, call_chain):
        for edge in self.output_edges:
            edge.node.consume(self.drop_to(edge), call_chain)
        for edge in self.output_edges:
            edge.node.step(call_chain)

    def drop_to(self, edge):
        return edge.value


class Pool(Node):
    EMPTY_INPUT = False
    EMPTY_OUTPUT = True
    MAX_INPUT = 2
    MAX_OUTPUT = 3
    COLOR = "skyblue"

    def __init__(self, name=None, id=None):
        super().__init__(name, id=id)
        self.pool = 0

    def step(self, call_chain):
        # check for loops and stop if loop detected
        if self in call_chain:
            return
        call_chain.append(self)
        for edge in self.output_edges:
            edge.node.consume(call_chain)

    def consume(self, value, call_chain):
        self.pool += value
        self.step(call_chain)

    def reset(self):
        self.pool = 0


class FixedPool(Pool):
    def __init__(self, name=None):
        super().__init__(name)

    def get_fix(self):
        return max([e.value for e in self.output_edges])

    def consume(self, value, call_chain):
        if not self.pool >= self.get_fix():
            self.pool += value
        self.step(call_chain)


class Converter(Node):
    EMPTY_INPUT = False
    EMPTY_OUTPUT = False
    MAX_INPUT = 3
    MAX_OUTPUT = 1
    COLOR = "gold"

    def __init__(self, name=None, id=None):
        super().__init__(name, id=id)
        self.called = False

    def consume(self, call_chain):
        if self.called is False:
            resources_available = True

            if isinstance(self.input_edges[0].node, RandomGate):  # TODO fix this ugly workaround
                self.output_edges[0].node.consume(self.output_edges[0].value, call_chain)
                self.called = True
                return

            for input_e in self.input_edges:
                if not input_e.node.pool >= input_e.value:
                    resources_available = False
            if resources_available is True:
                for input_e in self.input_edges:
                    input_e.node.pool -= input_e.value
                for output_e in self.output_edges:
                    output_e.node.consume(output_e.value, call_chain)
                self.called = True
            self.step(call_chain)


# class Trader(Node):
#     pass

class RandomGate(Node):
    EMPTY_INPUT = False
    EMPTY_OUTPUT = False
    MAX_INPUT = 1
    MAX_OUTPUT = 3
    COLOR = "red"

    def step(self, call_chain):
        for edge in self.output_edges:
            edge.node.step(call_chain)

    def consume(self, entity, call_chain):
        probs = [edge.value for edge in self.output_edges]
        for i in range(int(entity)):
            edge = np.random.choice(self.output_edges, 1, p=probs)[0]
            if isinstance(edge.node, Converter):
                edge.node.consume(call_chain)
            else:
                edge.node.consume(1, call_chain)
        self.step(call_chain)


class Drain(Pool):
    MAX_OUTPUT = 0
    COLOR = "darkorange"

    def consume(self, value=None, call_chain=None):
        for input_e in self.input_edges:
            # only drain resources if there are enough available
            if input_e.node.pool >= input_e.value:
                input_e.node.pool -= input_e.value
                self.pool += input_e.value


class Result(Pool):
    def step(self):
        pass


Source.ALLOWED_INPUT = []
Source.ALLOWED_OUTPUT = [Pool, FixedPool, RandomGate]
Pool.ALLOWED_INPUT = [Source, RandomGate, Converter]
Pool.ALLOWED_OUTPUT = [Converter, Drain]
Converter.ALLOWED_INPUT = [Pool, FixedPool, RandomGate]
Converter.ALLOWED_OUTPUT = [Pool, RandomGate]
RandomGate.ALLOWED_INPUT = [Source, Converter]
RandomGate.ALLOWED_OUTPUT = [Pool, Converter]
Drain.ALLOWED_INPUT = [Pool]
Drain.ALLOWED_OUTPUT = []
Result.ALLOWED_INPUT = [Converter]
Result.ALLOWED_OUTPUT = []
FixedPool.ALLOWED_INPUT = [Source, RandomGate, Converter]
FixedPool.ALLOWED_OUTPUT = [Converter, Drain]
