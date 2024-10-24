import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from geevo.nodes import RandomGate
from geevo.simulation import Simulator


class Graph:
    def __init__(self, config, edge_list, weights=None, plot_pos=None):
        self.config = config
        self.edge_list = edge_list
        if weights is not None:
            self.weights = weights[0]
            self.weights_prob = weights[1]
        else:
            self.weights = np.random.randint(1, 4, size=len(edge_list))
            self.weights_prob = np.random.dirichlet(np.ones(self.config[RandomGate] * 3), size=1)[0]
        self.nodes = self.init_nodes(names=True)
        self.simulator = None
        self.plot_pos = plot_pos

    def init_nodes(self, names=False):
        nodes = []
        count = 0
        for k, v in self.config.items():
            if names is False:
                nodes.extend([k() for _ in range(v)])
            else:
                for _ in range(v):
                    nodes.append(k(name=f"{k.__name__}-{count}"))
                    count += 1

        # if self.weights is None:
        #     self.weights = np.random.randint(1, 4, size=len(self.edge_list)).tolist()
        count = 0
        count_r = 0
        for edge in self.edge_list:
            if not isinstance(nodes[edge[0]], RandomGate):
                nodes[edge[0]].connect(nodes[edge[1]], self.weights[count])
                count += 1
            else:
                nodes[edge[0]].connect(nodes[edge[1]], self.weights_prob[count_r])
                count_r += 1

        # check probs of random gates
        for gate in [n for n in nodes if isinstance(n, RandomGate)]:
            edge_values = [e.value for e in gate.output_edges]
            values_sum = sum(edge_values)
            edge_values_scaled = [v / values_sum for v in edge_values]
            for e, v in zip(gate.output_edges, edge_values_scaled):
                e.value = v
        return nodes

    def save(self, file="graph.pkl"):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file="graph.pkl"):
        with open(file, "rb") as f:
            return pickle.load(f)

    def update_edge_weights(self, weights):
        self.weights = weights[0]
        self.weights_prob = weights[1]
        self.nodes = self.init_nodes(names=True)

    def update_edge_weights_random(self):
        self.weights = np.random.randint(1, 8, size=len(self.edge_list))
        self.weights_prob = np.random.dirichlet(np.ones(self.config[RandomGate] * 3), size=1)[0]
        self.nodes = self.init_nodes(names=True)

    def simulate(self, steps=50):
        self.simulator = Simulator(self.nodes)
        self.simulator.run(steps=steps)
        return self.simulator.monitoring

    def plot(self, figsize=(10, 4.5), save=False, filename="plots/graph.png", node_labels=None, edge_labels=None):
        g = nx.DiGraph()
        g.add_edges_from(sorted(self.edge_list))
        if self.plot_pos is None:
            try:
                pos = nx.planar_layout(g, scale=10)
            except nx.NetworkXException:
                print("Graph is probably not planar, so I choose spring layout for plotting.")
                pos = nx.spring_layout(g)
        else:
            pos = self.plot_pos
        plt.figure(figsize=figsize)

        if node_labels is None:
            node_labels = {idx: f"{type(node).__name__}-{idx}" for idx, node in enumerate(self.nodes)}
        node_colors = [self.nodes[idx].COLOR for idx in g.nodes]
        nx.draw(g, pos=pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors,
                font_color='black', font_size=10, labels=node_labels, arrows=True)

        if edge_labels is None:
            edge_labels = {}
            for i in range(len(self.nodes)):
                for o in self.nodes[i].output_edges:
                    edge_labels[(i, self.nodes.index(o.node))] = round(o.value, 2)
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.tight_layout()

        if save is True:
            plt.savefig(filename, dpi=300)
        plt.show()

    def get_nodes_of(self, node_type):
        return [n for n in self.nodes if isinstance(n, node_type)]


class Graph2:
    def __init__(self, config, edge_list, weights=None):
        self.config = config
        self.edge_list = edge_list
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randint(1, 10, size=len(edge_list))
        self.nodes = self.init_nodes(names=True)
        self.simulator = None

    def init_nodes(self, names=False):
        nodes = []
        count = 0
        for k, v in self.config.items():
            if names is False:
                nodes.extend([k() for _ in range(v)])
            else:
                for _ in range(v):
                    nodes.append(k(name=f"{k.__name__}-{count}", id=count))
                    count += 1

        for idx, edge in enumerate(self.edge_list):
            try:
                nodes[edge[0]].connect(nodes[edge[1]], self.weights[idx])
            except ValueError as e:
                print(e)
                print(edge)
                raise e

        # check probs of random gates
        for gate in [n for n in nodes if isinstance(n, RandomGate)]:
            edge_values = [e.value for e in gate.output_edges]
            values_sum = sum(edge_values)
            edge_values_scaled = [v / values_sum for v in edge_values]
            for e, v in zip(gate.output_edges, edge_values_scaled):
                e.value = v

        # for n in nodes:
        #     print(n)
        return nodes

    def save(self, file="graph.pkl"):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file="graph.pkl"):
        with open(file, "rb") as f:
            return pickle.load(f)

    def update_edge_weights(self, weights):
        self.weights = weights
        self.nodes = self.init_nodes(names=True)

    def update_edge_weights_random(self):
        self.weights = np.random.randint(1, 10, size=len(self.edge_list))
        self.nodes = self.init_nodes(names=True)

    def simulate(self, steps=50):
        self.simulator = Simulator(self.nodes)
        self.simulator.run(steps=steps)
        return self.simulator.monitoring

    def plot(self, figsize=(10, 4.5), save=False, filename="plots/graph.png", node_labels=None, edge_labels=None):
        g = nx.DiGraph()
        g.add_edges_from(sorted(self.edge_list))
        try:
            pos = nx.planar_layout(g, scale=10)
        except nx.NetworkXException:
            print("Graph is probably not planar, so I choose spring layout for plotting.")
            pos = nx.spring_layout(g)
        plt.figure(figsize=figsize)

        if node_labels is None:
            node_labels = {idx: f"{node.name}" for idx, node in enumerate(self.nodes)}
        node_colors = [self.nodes[idx].COLOR for idx in g.nodes]
        nx.draw(g, pos=pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors,
                font_color='black', font_size=10, labels=node_labels, arrows=True)

        if edge_labels is None:
            edge_labels = {}
            for i in range(len(self.nodes)):
                for o in self.nodes[i].output_edges:
                    edge_labels[(i, self.nodes.index(o.node))] = round(o.value, 2)
        print(edge_labels)
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.tight_layout()

        if save is True:
            plt.savefig(filename, dpi=300)
        plt.show()

    def get_nodes_of(self, node_type):
        return [n for n in self.nodes if isinstance(n, node_type)]


class Graph3(Graph2):
    # also supports State Connections and Registers
    def __init__(self, config, edge_list, registers, weights=None):
        super().__init__(config, edge_list, weights=weights)

        # connect registers with pools and edges
        self.registers = registers
        for r in self.registers:
            for v, inp in r.input_state_connection.items():
                inp.output_pool = self.nodes[inp.output_pool_id]
                inp.register_input = r
                # append register to fixed pool
                self.nodes[inp.output_pool_id].registers.append(r)
            for out in r.output_state_connection:
                out.register_output = r
                out.edge_input = self.nodes[out.edge_input_id[0]].get_edge_to(self.nodes[out.edge_input_id[1]])
                out.node_output = self.nodes[out.edge_input_id[0]]

    def simulate(self, steps=50):
        self.simulator = Simulator(self.nodes, self.registers)
        self.simulator.run(steps=steps)
        return self.simulator.monitoring



