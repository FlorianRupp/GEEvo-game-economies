from geevo.graph import Graph
from geevo.nodes import *


class Balancer:
    def __init__(self, config, edge_list, balance_pool_ids, pop_size=10, n_sim=10, n_sim_steps=100, frozen_weights=None,
                 balance_value=30, alpha=0.01):
        self.config = config
        self.edge_list = edge_list
        self.monitor = self.monitor = {"best": [], "avg": []}
        self.pop_size = pop_size
        self.population = []
        self.n_sim = n_sim
        self.n_sim_steps = n_sim_steps
        self.frozen_weights = frozen_weights
        self.init_population()
        self.result = None
        self.balance_pool_ids = balance_pool_ids
        self.balance_value = balance_value
        self.threshold = 1 - alpha

    def init_ind(self):
        g = Graph(config=self.config, edge_list=self.edge_list)
        edges = sum([len(n.output_edges) for n in g.get_nodes_of(RandomGate)])
        return [np.random.randint(1, 4, size=len(self.edge_list) - edges).tolist(),
                [round(i, 2) for i in np.random.uniform(0, 1, size=edges).tolist()]]

    def init_population(self):
        g = Graph(config=self.config, edge_list=self.edge_list)
        edges = sum([len(n.output_edges) for n in g.get_nodes_of(RandomGate)])
        for i in range(self.pop_size):
            self.population.append([np.random.randint(1, 8, size=len(self.edge_list) - edges).tolist(),
                                    [round(i, 2) for i in np.random.uniform(0, 1, size=edges).tolist()]])

    def get_ind_fitness_single(self, ind):
        # minimize diff of two pools
        try:
            g = Graph(config=self.config, edge_list=self.edge_list, weights=ind)
            res = g.simulate(self.n_sim_steps)
        except ZeroDivisionError:
            # most likely invalid probabilities (zero sum) for random gate, giving now a bad fitness
            return 0
        keys = list(res.keys())
        values = sorted([res[keys[self.balance_pool_ids[0]]][-1], res[keys[self.balance_pool_ids[1]]][-1]])
        try:
            return round(values[0] / values[1], 2)
        except ZeroDivisionError:
            return 1

    def get_ind_fitness_single2(self, ind):
        # minimize diff of all pools along regular time steps
        try:
            g = Graph(config=self.config, edge_list=self.edge_list, weights=ind)
            res = g.simulate(self.n_sim_steps)
        except ZeroDivisionError:
            # most likely invalid probabilities (zero sum) for random gate, giving now a bad fitness
            return 0
        keys = [p for p in res.keys() if not isinstance(p, Drain)]
        values = []
        for i in range(0, self.n_sim_steps, 10):
            values.append(sum([res[k][i] for k in keys]))
        res = []
        for idx in range(len(values) - 1):
            res.append(values[idx] - values[idx + 1])
        return sum(res) / len(res)

    def get_ind_fitness_single3(self, ind):
        # align a given pool toward a given absolute value
        try:
            g = Graph(config=self.config, edge_list=self.edge_list, weights=ind)
            res = g.simulate(self.n_sim_steps)
        except ZeroDivisionError:
            # most likely invalid probabilities (zero sum) for random gate, giving now a bad fitness
            return 0
        keys = list(res.keys())
        constraint = self.balance_value
        values = sorted([res[keys[self.balance_pool_ids[0]]][-1], constraint])
        try:
            return round(values[0] / values[1], 2)
        except ZeroDivisionError:
            return 1

    def get_ind_fitness(self, ind):
        res = [self.get_ind_fitness_single3(ind) for _ in range(self.n_sim)]
        return round(sum(res) / len(res), 2)

    def get_fitness(self, return_always=False):
        fitness = []
        for ind in self.population:
            fitness.append(self.get_ind_fitness(ind))

        # sort by fitness
        fitness_sorted = np.argsort(fitness)
        pop_ = np.array(self.population, dtype=object)
        fitness = np.array(fitness)
        pop_ = pop_[fitness_sorted][::-1][:self.pop_size]
        self.population = pop_.tolist()
        # print(sorted(fitness)[-1])
        self._monitor(fitness)
        if fitness.max() >= self.threshold:
            self.result = self.population[0]
            return fitness.max()
        if return_always is True:
            return fitness.max()

    def crossover(self):
        indices = list(range(len(self.population)))
        random.shuffle(indices)
        new = []
        for idx in range(len(indices))[::2]:
            one = self.population[indices[idx]]
            two = self.population[indices[idx + 1]]
            split_point = np.random.randint(max(len(one), 1))
            new.append([[*one[0][:split_point], *two[0][split_point:]], one[1]])
            new.append([[*one[0][:split_point], *two[0][split_point:]], two[1]])

            # cross probabilistic weights
            mean = np.mean(one[1]) * 0.2
            if np.random.randint(1) == 1:  # + or -
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(one[1]) + mean)).tolist()])
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(two[1]) + mean)).tolist()])
            else:
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(one[1]) - mean)).tolist()])
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(two[1]) - mean)).tolist()])
        self.population.extend(new)

    def mutate(self):
        selection = np.random.randint(len(self.population))
        selection_weight = np.random.randint(len(self.population[0][0]))
        mutation = np.random.randint(8)
        if np.random.randint(1) == 0:
            self.population[selection][0][selection_weight] += mutation
        else:
            self.population[selection][0][selection_weight] -= mutation
            if self.population[selection][0][selection_weight] < 1:
                self.population[selection][0][selection_weight] = 1

    def handle_frozen_weights(self):
        if self.frozen_weights is not None:
            for p in self.population:
                for i in self.frozen_weights:
                    p[0][i] = 1

    def run(self, steps=100):
        iterations = steps
        for i in range(steps):
            self.crossover()
            self.mutate()
            if i % 5 == 0:
                self.population.append(self.init_ind())
            self.handle_frozen_weights()
            fitness = self.get_fitness()
            if fitness is not None:
                print(f"Stopped after {i} iteration with a fitness of: {fitness}")
                iterations = i
                break
        if fitness is None:
            fitness = self.get_fitness(return_always=True)
        return fitness, iterations

    def _monitor(self, fitness):
        self.monitor["best"].append(fitness.max())
        self.monitor["avg"].append(fitness.mean())

    def plot_monitor(self):
        fig = plt.figure(figsize=(7, 5))
        for k, v in self.monitor.items():
            plt.plot(list(range(len(v))), v, label=k)
        plt.legend()


class BalancerV2:
    # fit on two different economies at once
    def __init__(self, g_config1, g_config2, pop_size=10, n_sim=10, n_sim_steps=100, threshold=0.99):
        self.g_config1 = g_config1
        self.g_config2 = g_config2
        self.monitor = self.monitor = {"best": [], "avg": []}
        self.pop_size = pop_size
        self.population = []
        self.n_sim = n_sim
        self.n_sim_steps = n_sim_steps
        self.init_population()
        self.result = None
        self.threshold = threshold
        print(n_sim)

    def init_ind(self):
        g1 = Graph(config=self.g_config1["conf"], edge_list=self.g_config1["edges"])
        g2 = Graph(config=self.g_config2["conf"], edge_list=self.g_config2["edges"])
        edges1 = sum([len(n.output_edges) for n in g1.get_nodes_of(RandomGate)])
        edges2 = sum([len(n.output_edges) for n in g2.get_nodes_of(RandomGate)])
        ind_g1 = [np.random.randint(1, 4, size=len(self.g_config1["edges"]) - edges1).tolist(),
                  [round(i, 2) for i in np.random.uniform(0, 1, size=edges1).tolist()]]
        ind_g2 = [np.random.randint(1, 4, size=len(self.g_config2["edges"]) - edges2).tolist(),
                  [round(i, 2) for i in np.random.uniform(0, 1, size=edges2).tolist()]]
        return [ind_g1, ind_g2]

    def init_population(self):
        for i in range(self.pop_size):
            self.population.append(self.init_ind())

    def get_ind_fitness_single(self, ind):
        # minimize diff of two pools of two diff graphs
        try:
            g1 = Graph(config=self.g_config1["conf"], edge_list=self.g_config1["edges"], weights=ind[0])
            res1 = g1.simulate(self.n_sim_steps)
            g2 = Graph(config=self.g_config2["conf"], edge_list=self.g_config2["edges"], weights=ind[1])
            res2 = g2.simulate(self.n_sim_steps)
        except ZeroDivisionError:
            # most likely invalid probabilities (zero sum) for random gate, giving now a bad fitness
            return 0
        keys1, keys2 = list(res1.keys()), list(res2.keys())
        values = sorted(
            [res1[keys1[self.g_config1["balance_node"]]][-1], res2[keys2[self.g_config2["balance_node"]]][-1]])
        try:
            return round(values[0] / values[1], 2)
        except ZeroDivisionError:
            return 1

    def get_ind_fitness(self, ind):
        res = [self.get_ind_fitness_single(ind) for _ in range(self.n_sim)]
        return round(sum(res) / len(res), 2)

    def get_fitness(self):
        fitness = []
        for ind in self.population:
            fitness.append(self.get_ind_fitness(ind))

        # sort by fitness
        fitness_sorted = np.argsort(fitness)
        pop_ = np.array(self.population, dtype=object)
        fitness = np.array(fitness)
        pop_ = pop_[fitness_sorted][::-1][:self.pop_size]
        self.population = pop_.tolist()
        print(sorted(fitness)[-1])
        self._monitor(fitness)
        if fitness.max() >= self.threshold:
            self.result = self.population[0]
            return fitness.max()

    def crossover(self):
        indices = list(range(len(self.population)))
        random.shuffle(indices)
        new = []
        for idx in range(len(indices))[::2]:
            one = self.population[indices[idx]]
            two = self.population[indices[idx + 1]]
            split_point1 = np.random.randint(max(len(one[0][0]), 1))
            split_point2 = np.random.randint(max(len(one[1][0]), 1))

            new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]], one[0][1]], one[1]])
            new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]], two[0][1]], one[1]])
            new.append([two[0], [[*one[1][0][:split_point2], *two[1][0][split_point2:]], one[1][1]]])
            new.append([two[0], [[*one[1][0][:split_point2], *two[1][0][split_point2:]], two[1][1]]])

            # cross probabilistic weights
            mean1 = np.mean(one[0][1]) * 0.2
            mean2 = np.mean(one[1][1]) * 0.2
            if np.random.randint(1) == 1:  # + or -
                new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
                             (abs(np.array(one[0][1]) + mean1)).tolist()], one[1]])
                new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
                             (abs(np.array(two[0][1]) + mean1)).tolist()], one[1]])
                new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
                                     (abs(np.array(one[1][1]) + mean2)).tolist()]])
                new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
                                     (abs(np.array(two[1][1]) + mean2)).tolist()]])
            else:
                new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
                             (abs(np.array(one[0][1]) - mean1)).tolist()], one[1]])
                new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
                             (abs(np.array(two[0][1]) - mean1)).tolist()], one[1]])
                new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
                                     (abs(np.array(one[1][1]) - mean2)).tolist()]])
                new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
                                     (abs(np.array(two[1][1]) - mean2)).tolist()]])
        self.population.extend(new)

    def mutate(self):
        def mutate_g(g):
            selection = np.random.randint(len(self.population))
            selection_weight = np.random.randint(len(self.population[0][g][0]))
            mutation = np.random.randint(8)
            if np.random.randint(1) == 0:
                self.population[selection][g][0][selection_weight] += mutation
            else:
                self.population[selection][g][0][selection_weight] -= mutation
                if self.population[selection][g][0][selection_weight] < 1:
                    self.population[selection][g][0][selection_weight] = 1

        mutate_g(0)
        mutate_g(1)

    def handle_frozen_weights(self):
        for p in self.population:
            for i in self.g_config1["frozen_weights"]:
                p[0][0][i] = 1
            for i in self.g_config2["frozen_weights"]:
                p[1][0][i] = 1

    def run(self, steps=100):
        for i in range(steps):
            self.crossover()
            self.mutate()
            self.handle_frozen_weights()
            fitness = self.get_fitness()
            if fitness is not None:
                print(f"Stopped after {i} iteration with a fitness of: {fitness}")
                break

    def _monitor(self, fitness):
        self.monitor["best"].append(fitness.max())
        self.monitor["avg"].append(fitness.mean())

    def plot_monitor(self):
        fig = plt.figure(figsize=(7, 5))
        for k, v in self.monitor.items():
            plt.plot(list(range(len(v))), v, label=k)
        plt.legend()
