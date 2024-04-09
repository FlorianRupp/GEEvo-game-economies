from datetime import datetime

import networkx as nx
import numpy as np
import random


class EvolutionaryGraphGeneration:
    def __init__(self, config, population_size=10):
        self.config = config
        self.population_size = population_size
        self.population = []
        self.init_pop()
        self.fitness = []
        self.iterations = 0

    def init_nodes(self):
        nodes = []
        count = 0
        for k, v in self.config.items():
            for _ in range(v):
                nodes.append(k(id=count))
                count += 1
        return nodes

    def init_pop(self):
        self.population = [self.init_nodes() for _ in range(self.population_size)]

    def connect_random(self):
        for ind in self.population:
            one = np.random.randint(len(ind))
            two = np.random.randint(len(ind))
            # print(f"Connect {nodes[one]} with {nodes[two]}")
            try:
                ind[one].connect(ind[two], 1, node_id=two)
            except ValueError:
                pass

    def delete_random(self):
        for ind in self.population:
            one = np.random.randint(len(ind))
            try:
                two = np.random.randint(len(ind[one].get_output_nodes()))
                ind[one].disconnect(ind[one].get_output_nodes()[two])
            except Exception:
                pass

    def get_edge_list(self, ind):
        edge_list = []
        for n in ind:
            edge_list.extend([(n.id, e.node_id) for e in n.output_edges])
        return edge_list

    def get_fitness(self):
        fitness_list = []
        for ind in self.population:
            fitness = sum([n.get_state() for n in ind]) * -1
            g = nx.Graph()
            g.add_edges_from(self.get_edge_list(ind))
            try:
                if not nx.is_connected(g):
                    fitness += 1
            except nx.NetworkXPointlessConcept:
                fitness += 1
            fitness_list.append(fitness)
        self.population = np.array(self.population)[np.argsort(fitness_list)].tolist()
        return sorted(fitness_list)[0]

    def crossover(self):
        indices = list(range(len(self.population)))
        random.shuffle(indices)
        if len(indices) % 2 == 1:
            indices = indices[:-1]
        for idx in range(len(indices)):
            one = self.get_edge_list(self.population[indices[idx]])
            two = self.get_edge_list(self.population[indices[idx + 1]])
            n = self.init_nodes()

            pos = np.random.randint(len(one))
            n[two[idx][0]].connect(n[one[idx][1]], 1, node_id=one[idx][1])

            for idx in range(sorted([len(one), len(two)])[0]):
                try:
                    if np.random.randint(1) == 0:
                        n[one[idx][0]].connect(n[two[idx][1]], 1, node_id=two[idx][1])
                    else:
                        n[two[idx][0]].connect(n[one[idx][1]], 1, node_id=one[idx][1])
                except ValueError:
                    # do not connect if connection is not allowed
                    pass
            self.population.append(n)

    def run(self, steps=150_000):
        start = datetime.now()
        for i in range(steps):
            self.connect_random()
            if i % 10 == 0:
                self.delete_random()
            # self.crossover()
            self.fitness.append(self.get_fitness())
            if self.fitness[-1] == 0:
                end = datetime.now()
                print(
                    f"Stopped after {i} iterations in {(end - start).microseconds / 1000}ms. Num edges {len(self.get_edge_list(self.population[0]))}, fitness: {self.fitness[-1]}")
                self.iterations = i
                return True
        print(
            f"Time exceeded, stopped after {i} iterations. Num edges {len(self.get_edge_list(self.population[0]))}, fitness: {self.fitness[-1]}")
        return False

    def get_best(self):
        # only call after run has successfully been executed
        return self.get_edge_list(self.population[0])
