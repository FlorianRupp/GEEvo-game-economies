import matplotlib.pyplot as plt
from geevo.graph import Graph
from geevo.nodes import *


class Balancer:
    """
    Algoritma Genetika (Genetic Algorithm) untuk optimasi bobot graf (Edge Weights Optimizer).
    Objektif: Mencarikan konfigurasi ruang matriks W sedemikian rupa sehingga grafik aliran simulasi GEEvo 
    mencapai konvergensi (memaksimalkan atau menyeimbangkan nilai resource di Pool tertentu)
    berdasarkan definisi fungsi Fitness f(w).
    """
    def __init__(self, config, edge_list, balance_pool_ids, pop_size=10, n_sim=10, n_sim_steps=100, frozen_weights=None,
                 balance_value=30, alpha=0.01, agent=None, fitness_func="obj3"):
        self.config = config
        self.edge_list = edge_list
        # Pencatatan skor metrik statistik populasi f_max dan f_mean
        self.monitor = {"best": [], "avg": []}
        
        # M: Skalar kapasitas individu dalam satu generasi reproduksi (Populasi)
        self.pop_size = pop_size
        self.population = []
        
        # N_sim: Jumlah eksekusi Monte-Carlo untuk mengukur rata-rata ekspektasi stokastik
        self.n_sim = n_sim
        self.n_sim_steps = n_sim_steps
        
        # Array/Set Constraint khusus (Tetapan konstanta tepi graf yang dilarang bermutasi)
        self.frozen_weights = frozen_weights
        self.init_population()
        self.result = None
        
        # Obyektif optimasi
        self.balance_pool_ids = balance_pool_ids
        self.balance_value = balance_value # Nilai target E(X) konstrain
        self.alpha = alpha                 # Toleransi deviasi (Misal \alpha = 0.05)
        self.threshold = 1 - alpha         # Batas Limit / Tolerance Tingkat Konvergensi (e Misal \approx 0.99)
        self.agent = agent
        self.fitness_func = fitness_func   # Memilih formula evaluasi (default: 'obj3', baru: 'obj4')
        
        self.g = Graph(config=self.config, edge_list=self.edge_list)
        # Setting awal konverter (Menghapus kontrol auto-loop fungsi transformasi iteratif)
        for c in self.g.get_nodes_of(Converter):
            c.is_auto = False
            
        # Alokasi matriks Memoisasi (Dynamic Programming/Cache) menghindari perhitungan f(x) redundan
        self._fit_cache = {}

    def init_ind(self):
        # Pembentukan 1 Indivdu: Tuple dari [vektor skalar bobot deterministik, vektor probabilitas p_1...p_k]
        g = Graph(config=self.config, edge_list=self.edge_list)
        # Menghitung kardinalitas total output percabangan Markov
        edges = sum([len(n.output_edges) for n in g.get_nodes_of(RandomGate)])
        
        # Mengembalikan matriks bobot acak seragam diskrit \in [1, 3] dan probabilitas seragam riil \in [0.0, 1.0)
        return [np.random.randint(1, 4, size=len(self.edge_list) - edges).tolist(),
                [round(i, 2) for i in np.random.uniform(0, 1, size=edges).tolist()]]

    def init_population(self):
        # Menciptakan Himpunan (Populasi Basis 0) sebanyak self.pop_size
        g = Graph(config=self.config, edge_list=self.edge_list)
        edges = sum([len(n.output_edges) for n in g.get_nodes_of(RandomGate)])
        for i in range(self.pop_size):
            self.population.append([np.random.randint(1, 8, size=len(self.edge_list) - edges).tolist(),
                                    [round(i, 2) for i in np.random.uniform(0, 1, size=edges).tolist()]])

    def get_ind_fitness_single3(self, ind):
        """
        Fungsi Objektif 3: Mencari konvergensi terhadap satu nilai titik tumpu (L)
        f = \frac{\min(x, L)}{\max(x, L)} \to \max (\approx 1.0)
        """
        agents = self.agent if isinstance(self.agent, list) else [self.agent]
        fitnesses = []
        self.g.set_edge_weights(ind)
        for agent in agents:
            for n in self.g.nodes:
                if hasattr(n, 'called'): n.called = False
            try:
                res = self.g.simulate(self.n_sim_steps, agent=agent)
            except ZeroDivisionError:
                fitnesses.append(0)
                continue
                
            keys = list(res.keys())
            constraint = self.balance_value
            # Memilah Minimum dan Maksimum penyebut pecahan (x \le y)
            values = sorted([res[keys[self.balance_pool_ids[0]]][-1], constraint])
            try:
                fitnesses.append(round(values[0] / values[1], 2))
            except ZeroDivisionError:
                fitnesses.append(1)
        return round(sum(fitnesses) / len(fitnesses), 2) if fitnesses else 0

    def get_ind_fitness_single4(self, ind):
        """
        Fungsi Objektif 4 (Eq 2 Paper): Mengevaluasi konvergensi nilai aktual (s_t) suatu Pool 
        di langkah waktu akhir t (\sim n_sim_steps) terhadap target absolut (x) dengan ambang toleransi \alpha.
        """
        agents = self.agent if isinstance(self.agent, list) else [self.agent]
        fitnesses = []
        self.g.set_edge_weights(ind)
        for agent in agents:
            for n in self.g.nodes:
                if hasattr(n, 'called'): n.called = False
            try:
                res = self.g.simulate(self.n_sim_steps, agent=agent)
            except ZeroDivisionError:
                fitnesses.append(0)
                continue
                
            keys = list(res.keys())
            target_x = self.balance_value
            # s_t: Aktual Pool value di langkah waktu t akhir
            s_t = res[keys[self.balance_pool_ids[0]]][-1]
            
            # Aplikasi Toleransi Berbasis Alpha: |x - s_t| <= alpha * x
            if target_x == 0:
                f_t = 1.0 if s_t == 0 else 0.0
            else:
                if abs(target_x - s_t) <= self.alpha * target_x:
                    f_t = 1.0
                else:
                    f_t = min(s_t, target_x) / max(s_t, target_x)
            fitnesses.append(round(f_t, 2))
            
        return round(sum(fitnesses) / len(fitnesses), 2) if fitnesses else 0

    def get_ind_fitness(self, ind):
        # Eksekusi Monte-Carlo untuk mengurangi noise bias variansi perilaku agen stokastik
        if self.fitness_func == "obj4":
            res = [self.get_ind_fitness_single4(ind) for _ in range(self.n_sim)]
        else:
            res = [self.get_ind_fitness_single3(ind) for _ in range(self.n_sim)]
            
        # Rataan Nilai Harapan \mu = \frac{1}{N} \sum_{i} x_i
        return round(sum(res) / len(res), 2)

    def get_fitness(self, return_always=False):
        # Agregat Fitness: Iterasi pemetaan memoisasi O(1) Cache Lookup mapping fungsi Fitness untuk seluruh Populasi
        fitness = []
        for ind in self.population:
            k = str(ind)
            if k not in self._fit_cache:
                self._fit_cache[k] = self.get_ind_fitness(ind)
            fitness.append(self._fit_cache[k])

        # Operasi "Survival of the fittest" selection. Array Sort secara Asimptotik O(N \log N)
        fitness_sorted = np.argsort(fitness)
        pop_ = np.array(self.population, dtype=object)
        fitness = np.array(fitness)
        
        # Pengirisan (Slicing) Array \to Mengeliminasi sebagian sampel terburuk (Ekor bawah distribusi fitness)
        # Menghapus dari indeks 0 hingga membiaskan \max \{N\}, urutan descending ([::-1])
        pop_ = pop_[fitness_sorted][::-1][:self.pop_size]
        self.population = pop_.tolist()

        # Update Catatan Epoch
        self._monitor(fitness)
        
        # Evaluasi Konvergensi epsilon
        if fitness.max() >= self.threshold:
            self.result = self.population[0]
            return fitness.max()
        if return_always is True:
            return fitness.max()

    def crossover(self):
        """Operasi pindah silang partisi batas array matriks koefisien (Crossover Algebra)"""
        indices = list(range(len(self.population)))
        random.shuffle(indices)
        new = []
        # Berpasangan step 2 individu untuk pembentukan Hibrida (Gen A x Gen B)
        for idx in range(len(indices))[::2]:
            one = self.population[indices[idx]]
            two = self.population[indices[idx + 1]]
            
            # Pemisahan Titik Tengah Random (Single-point crossover) dimensi diskrit
            split_point = np.random.randint(max(len(one), 1))
            
            # Kombinasi Silang Bagian Aliran Deterministik A-B dan Probabilistik (p_1..p_n) dari single parent tetua
            new.append([[*one[0][:split_point], *two[0][split_point:]], one[1]])
            new.append([[*one[0][:split_point], *two[0][split_point:]], two[1]])

            # Pembentukan Mutasi Crossover untuk Bobot Peluang Markov (Normalisasi aditif)
            # Menambah atau mengurangi vektor probabilitas dengan porsi drift (rata-rata * 0.2 koefisien volatilitas)
            mean = np.mean(one[1]) * 0.2
            if np.random.randint(1) == 1:  # Modus penambahan komutatif
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(one[1]) + mean)).tolist()])
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(two[1]) + mean)).tolist()])
            else:
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(one[1]) - mean)).tolist()])
                new.append([[*one[0][:split_point], *two[0][split_point:]], (abs(np.array(two[1]) - mean)).tolist()])
        
        # Menyatukan kembali dimensi individu anakan G_{n} \cup G_{n+1}
        self.population.extend(new)

    def mutate(self):
        """Random Edge Weight Mutasi. Merubah Vektor Elemen W secara deterministik \pm \Delta \sigma."""
        selection = np.random.randint(len(self.population))
        selection_weight = np.random.randint(len(self.population[0][0]))
        mutation = np.random.randint(8)  # Nilai Drift Skalar \sigma
        
        # Probabilitas Binary P = 0.5 Increment/Decrement
        if np.random.randint(1) == 0:
            self.population[selection][0][selection_weight] += mutation
        else:
            self.population[selection][0][selection_weight] -= mutation
            # Restriksi Hukum Inversitif (Non-Negative weights mapping > 0)
            if self.population[selection][0][selection_weight] < 1:
                self.population[selection][0][selection_weight] = 1

    def handle_frozen_weights(self):
        # Transformasi Constraint Variabel Konstan Paksa (Locking Weights $w_c = 1$)
        if self.frozen_weights is not None:
            for p in self.population:
                for i in self.frozen_weights:
                    p[0][i] = 1

    def run(self, steps=100):
        # Eksekusi Epoch Perulangan Evaluasi Genetika
        iterations = steps
        for i in range(steps):
            self.crossover()
            self.mutate()
            # Pembersihan lokal genetik ekstrim stagnasi: Memasukkan Gen baru ke ruang sampel acak setiap 5 Episentrum t
            if i % 5 == 0:
                self.population.append(self.init_ind())
                
            self.handle_frozen_weights()
            
            # Mendapatkan fungsi Objektif
            fitness = self.get_fitness()
            if fitness is not None:
                # Kriteria Konvergensi Sukses Tercapai F \to Optimasi Epsilon
                print(f"Stopped after {i} iteration with a fitness of: {fitness}")
                iterations = i
                break
                
        # Konvergensi Limit Waktu Penuh Gagal Terkoneksi Maksimal        
        if fitness is None:
            fitness = self.get_fitness(return_always=True)
        return fitness, iterations

    def _monitor(self, fitness):
        self.monitor["best"].append(fitness.max())
        self.monitor["avg"].append(fitness.mean())

    def plot_monitor(self):
        # Plotting Koordinat Bidang XY (Iterasi \to Fitness) Matplotlib
        fig = plt.figure(figsize=(7, 5))
        for k, v in self.monitor.items():
            plt.plot(list(range(len(v))), v, label=k)
        plt.legend()


# class BalancerV2:
#     # Implementasi f2 dari paper GEEvo: optimasi dua ekonomi game sekaligus (G1 x G2).
#     # Digunakan pada case study mage vs archer (Section IV-B2).
#     # Tidak dipakai dalam scope TA ini (hanya satu ekonomi yang dioptimasi).
#     def __init__(self, g_config1, g_config2, pop_size=10, n_sim=10, n_sim_steps=100, threshold=0.99, agent1=None, agent2=None):
#         self.g_config1 = g_config1
#         self.g_config2 = g_config2
#         self.monitor = {"best": [], "avg": []}
#         self.pop_size = pop_size
#         self.population = []
#         self.n_sim = n_sim
#         self.n_sim_steps = n_sim_steps
#         self.init_population()
#         self.result = None
#         self.threshold = threshold
#         self.agent1 = agent1
#         self.agent2 = agent2
#         print(n_sim)
#
#     def init_ind(self):
#         # Membentuk array tupel ganda dari sistem Graf 1 dan Graf 2
#         g1 = Graph(config=self.g_config1["conf"], edge_list=self.g_config1["edges"])
#         g2 = Graph(config=self.g_config2["conf"], edge_list=self.g_config2["edges"])
#         edges1 = sum([len(n.output_edges) for n in g1.get_nodes_of(RandomGate)])
#         edges2 = sum([len(n.output_edges) for n in g2.get_nodes_of(RandomGate)])
#         ind_g1 = [np.random.randint(1, 4, size=len(self.g_config1["edges"]) - edges1).tolist(),
#                   [round(i, 2) for i in np.random.uniform(0, 1, size=edges1).tolist()]]
#         ind_g2 = [np.random.randint(1, 4, size=len(self.g_config2["edges"]) - edges2).tolist(),
#                   [round(i, 2) for i in np.random.uniform(0, 1, size=edges2).tolist()]]
#         return [ind_g1, ind_g2]
#
#     def init_population(self):
#         for i in range(self.pop_size):
#             self.population.append(self.init_ind())
#
#     def get_ind_fitness_single(self, ind):
#         # Pengukuran Fungsi Loss: f(W_1, W_2) \to \mathbb{R} (Meminimalkan delta rasio skalar antar G1 Node dan G2 Node)
#         try:
#             g1 = Graph(config=self.g_config1["conf"], edge_list=self.g_config1["edges"], weights=ind[0])
#             res1 = g1.simulate(self.n_sim_steps, agent=self.agent1)
#             g2 = Graph(config=self.g_config2["conf"], edge_list=self.g_config2["edges"], weights=ind[1])
#             res2 = g2.simulate(self.n_sim_steps, agent=self.agent2)
#         except ZeroDivisionError:
#             # most likely invalid probabilities (zero sum) for random gate, giving now a bad fitness
#             return 0
#
#         keys1, keys2 = list(res1.keys()), list(res2.keys())
#         values = sorted(
#             [res1[keys1[self.g_config1["balance_node"]]][-1], res2[keys2[self.g_config2["balance_node"]]][-1]])
#         try:
#             return round(values[0] / values[1], 2)
#         except ZeroDivisionError:
#             return 1
#
#     def get_ind_fitness(self, ind):
#         res = [self.get_ind_fitness_single(ind) for _ in range(self.n_sim)]
#         return round(sum(res) / len(res), 2)
#
#     def get_fitness(self):
#         fitness = []
#         for ind in self.population:
#             fitness.append(self.get_ind_fitness(ind))
#
#         # Sort Sort \mathcal{O}(N\log N) Fitness
#         fitness_sorted = np.argsort(fitness)
#         pop_ = np.array(self.population, dtype=object)
#         fitness = np.array(fitness)
#         pop_ = pop_[fitness_sorted][::-1][:self.pop_size]
#         self.population = pop_.tolist()
#
#         print(sorted(fitness)[-1]) # Logging Ekspektasi Maksimum Global f \to \max (\mathbb{X})
#         self._monitor(fitness)
#         if fitness.max() >= self.threshold:
#             self.result = self.population[0]
#             return fitness.max()
#
#     def crossover(self):
#         # Replikasi genetik persilangan dengan matriks multi-dimensi (Sub-vektor parent A1, A2, B1, B2)
#         indices = list(range(len(self.population)))
#         random.shuffle(indices)
#         new = []
#         for idx in range(len(indices))[::2]:
#             one = self.population[indices[idx]]
#             two = self.population[indices[idx + 1]]
#             split_point1 = np.random.randint(max(len(one[0][0]), 1))
#             split_point2 = np.random.randint(max(len(one[1][0]), 1))
#
#             new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]], one[0][1]], one[1]])
#             new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]], two[0][1]], one[1]])
#             new.append([two[0], [[*one[1][0][:split_point2], *two[1][0][split_point2:]], one[1][1]]])
#             new.append([two[0], [[*one[1][0][:split_point2], *two[1][0][split_point2:]], two[1][1]]])
#
#             # Operasi Modulasi Properti Peluang / Normalisasi Mutasi p_i - Variabel Skalar Deviasi \sigma_x
#             mean1 = np.mean(one[0][1]) * 0.2
#             mean2 = np.mean(one[1][1]) * 0.2
#             if np.random.randint(1) == 1:  # + or -
#                 new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
#                              (abs(np.array(one[0][1]) + mean1)).tolist()], one[1]])
#                 new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
#                              (abs(np.array(two[0][1]) + mean1)).tolist()], one[1]])
#                 new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
#                                      (abs(np.array(one[1][1]) + mean2)).tolist()]])
#                 new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
#                                      (abs(np.array(two[1][1]) + mean2)).tolist()]])
#             else:
#                 new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
#                              (abs(np.array(one[0][1]) - mean1)).tolist()], one[1]])
#                 new.append([[[*one[0][0][:split_point1], *two[0][0][split_point1:]],
#                              (abs(np.array(two[0][1]) - mean1)).tolist()], one[1]])
#                 new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
#                                      (abs(np.array(one[1][1]) - mean2)).tolist()]])
#                 new.append([two[0], [[*one[1][0][:split_point1], *two[1][0][split_point1:]],
#                                      (abs(np.array(two[1][1]) - mean2)).tolist()]])
#         self.population.extend(new)
#
#     def mutate(self):
#         # Fungsi lambda internal / sub-fungsi untuk modularisasi per blok sub-genetik
#         def mutate_g(g):
#             selection = np.random.randint(len(self.population))
#             selection_weight = np.random.randint(len(self.population[0][g][0]))
#             mutation = np.random.randint(8)
#             if np.random.randint(1) == 0:
#                 self.population[selection][g][0][selection_weight] += mutation
#             else:
#                 self.population[selection][g][0][selection_weight] -= mutation
#                 if self.population[selection][g][0][selection_weight] < 1:
#                     self.population[selection][g][0][selection_weight] = 1
#
#         mutate_g(0)
#         mutate_g(1)
#
#     def handle_frozen_weights(self):
#         for p in self.population:
#             for i in self.g_config1["frozen_weights"]:
#                 p[0][0][i] = 1
#             for i in self.g_config2["frozen_weights"]:
#                 p[1][0][i] = 1
#
#     def run(self, steps=100):
#         # Loop Epochs
#         for i in range(steps):
#             self.crossover()
#             self.mutate()
#             self.handle_frozen_weights()
#             fitness = self.get_fitness()
#             if fitness is not None:
#                 print(f"Stopped after {i} iteration with a fitness of: {fitness}")
#                 break
#
#     def _monitor(self, fitness):
#         self.monitor["best"].append(fitness.max())
#         self.monitor["avg"].append(fitness.mean())
#
#     def plot_monitor(self):
#         fig = plt.figure(figsize=(7, 5))
#         for k, v in self.monitor.items():
#             plt.plot(list(range(len(v))), v, label=k)
#         plt.legend()
