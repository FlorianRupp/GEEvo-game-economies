from datetime import datetime

import networkx as nx
import numpy as np
import random


class EvolutionaryGraphGeneration:
    """
    Kelas pembuat model algoritma heuristik genetik/evolusi yang bertanggung jawab atas 
    generasi penciptaan matriks kejadian graf probabilitas. Ini adalah mesin optimasi diskrit pencarian acak 
    dengan prinsip 'Survival of the Fittest'.
    """
    def __init__(self, config, population_size=10):
        # `config`: Set Parameter Topologi Inisiasi Graf
        self.config = config
        
        # N = Ukuran Populasi (Jumlah skalar individu dalam satu dimensi ruang vektor kandidat generasi graf)
        self.population_size = population_size
        self.population = []
        # Mengisi array self.population dengan invididu ruang state setara (Gen Initialisasi Matriks N Kandidat)
        self.init_pop()
        
        # Larik (Vektor) penyimpanan himpunan historis nilai fitness terbaik
        self.fitness = []
        self.iterations = 0

    def init_nodes(self):
        # Pembentukan ruang basis state: Membuat array berisi Obyek Verteks (Nodes) 
        # dalam kalkulasi skalar dimensi \sum_{v=V_j} N(v). Total ukuran Node menjadi batas dimensi operasi Vektor G. 
        nodes = []
        count = 0
        # \forall k \in \text{config} \ldots
        for k, v in self.config.items():
            for _ in range(v):
                nodes.append(k(id=count))
                count += 1
        return nodes

    def init_pop(self):
        # Membuat ruang vektor \mathcal{P} kandidat graf \vec{G} sejumlah populasi dimensi M (N_pop)
        self.population = [self.init_nodes() for _ in range(self.population_size)]

    def connect_random(self):
        """
        Operasi Koneksi Graf Random (Edge Injection / Mutasi Topologi)
        Mengacak perantara (sisi berarah / directed edge) (u, v) secara probabilitas independen bernuansa seragam (Uniform)
        """
        for ind in self.population:
            # Mengambil 2 index bilangan cacah diskrit pseudo-random (independen) dari seragam (0 \dots |ind| - 1)
            one = np.random.randint(len(ind))
            two = np.random.randint(len(ind))
            # Proses injeksi \rightarrow menambahkan elemen ke dalam himpunan sisi E: E \cup (one, two)
            try:
                ind[one].connect(ind[two], 1, node_id=two)
            except ValueError:
                # Gagal (misal \exists constraint topologi, aturan edge diskoneksi) \Rightarrow Pass heuristik
                pass

    def delete_random(self):
        """
        Operasi Penghapusan Ruang Sisi Random (Edge Pruning Mutation / Mutasi Hapus Terarah)
        """
        for ind in self.population:
            # Sama seperti connect_random, mensampel verteks sumber secara stokastik pseudo-random seragam
            one = np.random.randint(len(ind))
            try:
                # Mengambil kardinalitas himpunan himpunan sisi luar E_{out} untuk titik `one`
                # kemudian mensampel indeks diskrit dan menghapus koneksi: e \notin E_{out}
                two = np.random.randint(len(ind[one].get_output_nodes()))
                ind[one].disconnect(ind[one].get_output_nodes()[two])
            except Exception:
                pass

    def get_edge_list(self, ind):
        # Proyektor Graf ke List of Tupel (u, v) -- representasi matematis G = (V, E)
        edge_list = []
        # Operasi skalar dari himpunan titik V: iterasi memetakan seluruh tuple pasangan node terhubung \in E
        for n in ind:
            edge_list.extend([(n.id, e.node_id) for e in n.output_edges])
        return edge_list

    def get_fitness(self):
        """
        Kalkulasi Objective Function (Loss / Fitness Function) f(\vec{G}) untuk mengevaluasi parameter optimal
        skor (angka riil skalar) suatu kandidat individu. Tujuan program ini umumnya adalah maksimasi/minimasi $f$.
        """
        fitness_list = []
        for ind in self.population:
            # Menghitung fitness dasar (Pelepasan state Node): Inisialisasi sigma metrik P=V (\sum -1 \cdot f_{\text{state}})
            fitness = sum([n.get_state() for n in ind]) * -1
            
            # Pemodelan matematis untuk Graph Konektivitas dengan pustaka networkx
            g = nx.Graph()
            # Pemetaan himpunan array E -- Sisi tidak berarah / Undirected Edge
            g.add_edges_from(self.get_edge_list(ind))
            try:
                # Constraint Tambahan \rightarrow Menghitung penalti jika graf diskonektif / terpisah 
                # (ada sub-graf komponen terpisah, melanggar relasi transitivitas konektivitas C \cong G)
                if not nx.is_connected(g):
                    fitness += 1
            except nx.NetworkXPointlessConcept:
                fitness += 1
                
            fitness_list.append(fitness)
            
        # Algoritma Pengurutan (Sorting) M=n\log n:  
        # Melakukan indeks partisi array menggunakan argmin (penyortiran nilai riil / float ke urutan terkecil)
        # Artinya fungsi fitness menseleksi nilai-nilai individu minimum.
        self.population = np.array(self.population)[np.argsort(fitness_list)].tolist()
        # Mengembalikan skalar penimbang Fitness (batas infimum himpunan array sorted - elemen paling optimal [0])
        return sorted(fitness_list)[0]

    def run(self, steps=150_000):
        # Eksekusi iterasi berbatas steps (Langkah komputasi iterasi batas epoch N)
        start = datetime.now()
        for i in range(steps):
            # Eksplorasi Stokastik
            self.connect_random()
            
            # Sub-seleksi (Kondisional Operasi Modulo periodisitas per t=10)
            if i % 10 == 0:
                self.delete_random()
                
            # self.crossover() -> Dimatikan/Komentar : Memodifikasi distribusi probabilistik ruang sampling (exploration-only)
            
            # Pencatatan skor
            self.fitness.append(self.get_fitness())
            
            # Batas Kondisi Akhir Henti Operasi Mutlak (Konvergen): Jika Fitness Skalar = 0
            if self.fitness[-1] == 0:
                end = datetime.now()
                # Pencetak waktu respon simulasi (\Delta t microdetik real time processing / 1000 = MS)
                print(
                    f"Stopped after {i} iterations in {(end - start).microseconds / 1000}ms. Num edges {len(self.get_edge_list(self.population[0]))}, fitness: {self.fitness[-1]}")
                self.iterations = i
                return True
                
        # Return limit akhir sub-optimal (Iterasi batas maksimum/habis)
        print(
            f"Time exceeded, stopped after {i} iterations. Num edges {len(self.get_edge_list(self.population[0]))}, fitness: {self.fitness[-1]}")
        return False

    def get_best(self):
        # Fungsi return pemanggilan skalar array indeks 0.
        # Konvensi the "survival of the fittest": Urutan array dipartisi oleh self.get_fitness beradasarkan indeks skor terkecil
        # only call after run has successfully been executed
        return self.get_edge_list(self.population[0])
