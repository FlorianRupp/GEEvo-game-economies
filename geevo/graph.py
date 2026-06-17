import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from geevo.nodes import RandomGate
from geevo.simulation import Simulator


class Graph:
    r"""
    Struktur Data Graf G = (V, E, W): Gabungan set titik (Simpul/Nodes) dan koneksi garis (Sisi/Edges).
    Ini merepresentasikan Jaringan Logika Aliran Dinamika Skalar dari interaksi matematis titik simulasi GEEvo.
    """
    def __init__(self, config, edge_list, weights=None, plot_pos=None):
        # Konfigurasi himpunan kuantitas tiap tipe titik
        self.config = config
        # Adjacency list: Koleksi array tupel E \subseteq V \times V merepresentasikan koneksi relasional A \to B
        self.edge_list = edge_list
        
        # Pengkondisian Vektor Bobot (Array Skalar Probabilitas/Koefisien Transmisi):
        if weights is not None:
            # Menggunakan bobot riil skalar yang diberikan w \in W
            self.weights = weights[0]
            # Probabilitas khusus Markov Gate p \in W_p (Sub-vektor)
            self.weights_prob = weights[1]
        else:
            # Pembuatan koefisien stokastik deterministik uniform U[1, 3] untuk inisialisasi awal.
            self.weights = np.random.randint(1, 4, size=len(edge_list))
            # Parameter Dirichlet Distribution (Distribusi Probabilitas Simplex n-dimensi).
            # Pemilihan acak sekumpulan nilai positif multivariat p_i yang selalu memenuhi kriteria jumlah total integral \sum p = 1
            self.weights_prob = np.random.dirichlet(np.ones(self.config[RandomGate] * 3), size=1)[0]
            
        # Mengeksekusi pembentukan Obyek Node berdasarkan topologi config awal \to Instansiasi Matriks Node V
        self.nodes = self.init_nodes(names=True)
        self.simulator = None
        # Vektor koordinat dimensi \mathbb{R}^2 bagi algoritma visual (Layout rendering posisi titik graf)
        self.plot_pos = plot_pos

    def init_nodes(self, names=False):
        """Memetakan (Membangun) array instansiasi simpul (set V) berdasar jumlah frekuensi jenis klasifikasi simpul."""
        nodes = []
        count = 0
        for k, v in self.config.items():
            if names is False:
                nodes.extend([k() for _ in range(v)])
            else:
                for _ in range(v):
                    # Injeksi string ID dan penghitung dimensi indeks enumerasi V_c
                    nodes.append(k(name=f"{k.__name__}-{count}"))
                    count += 1

        # Variabel inkremental indeks ruang state skalar (Menghitung skalar terasosiasi dengan edge)
        count = 0
        count_r = 0
        for edge in self.edge_list:
            # Modifikasi state pembentukan nilai Sisi: Edge Injection
            if not isinstance(nodes[edge[0]], RandomGate):
                # Bobot mutlak f(x)
                nodes[edge[0]].connect(nodes[edge[1]], self.weights[count])
                count += 1
            else:
                # Koefisien Probabilitas P(x)
                nodes[edge[0]].connect(nodes[edge[1]], self.weights_prob[count_r])
                count_r += 1

        # Verifikasi Probabilitas Simplex (Scaling/Normalisation step)
        # Menjamin properti fungsi probabilitas titik acak tetap valid (\sum p = 1.0) dengan perbandingan normal: v_i / \sum v
        for gate in [n for n in nodes if isinstance(n, RandomGate)]:
            edge_values = [e.value for e in gate.output_edges]
            values_sum = sum(edge_values)
            if values_sum > 0:
                edge_values_scaled = [v / values_sum for v in edge_values]
                for e, v in zip(gate.output_edges, edge_values_scaled):
                    e.value = v
        return nodes

    def save(self, file="graph.pkl"):
        # Menyimpan struktur matriks graf (Byte serialization array ke memori persisten komputer)
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file="graph.pkl"):
        # Deserialisasi dari file biner persisten (Rebuilding binary tree array memori komputer)
        with open(file, "rb") as f:
            return pickle.load(f)

    def update_edge_weights(self, weights):
        # Pembaharuan nilai vektor koefisien pada Adjacency List (digunakan saat fase Genetic Optimization Re-evaluation).
        # Substitusi W dengan state W_{\text{new}} \Rightarrow rebuild topologi graf (\mathcal{O}(V + E))
        self.weights = weights[0]
        self.weights_prob = weights[1]
        self.nodes = self.init_nodes(names=True)

    def set_edge_weights(self, weights):
        # Operasi pertukaran matriks bobot O(E) In-Place mutasi memori (Tanpa reinstansiasi graf total)
        self.weights = weights[0]
        self.weights_prob = weights[1]
        count = 0
        count_r = 0
        for edge in self.edge_list:
            u, v = edge[0], edge[1]
            if not isinstance(self.nodes[u], RandomGate):
                for e in self.nodes[u].output_edges:
                    if e.node == self.nodes[v]:
                        e.value = self.weights[count]  # Mutasi memori nilai skalar langsung ke obyek Sisi (x \mapsto v)
                        break
                count += 1
            else:
                for e in self.nodes[u].output_edges:
                    if e.node == self.nodes[v]:
                        e.value = self.weights_prob[count_r]
                        break
                count_r += 1

        # Rekonstruksi basis ekuivalensi (Normalisasi) properti \sum p_i = 1 kepada vertex RandomGate
        for gate in [n for n in self.nodes if isinstance(n, RandomGate)]:
            edge_values = [e.value for e in gate.output_edges]
            values_sum = sum(edge_values)
            if values_sum > 0:
                edge_values_scaled = [v / values_sum for v in edge_values]
                for e, v in zip(gate.output_edges, edge_values_scaled):
                    e.value = v

    def update_edge_weights_random(self):
        # Randomisasi vektor laju distribusi uniform (Bilangan riil 1 \le x < 8) dan skalar stokastik probabilistik p_i (Dirichlet)
        self.weights = np.random.randint(1, 8, size=len(self.edge_list))
        self.weights_prob = np.random.dirichlet(np.ones(self.config[RandomGate] * 3), size=1)[0]
        self.nodes = self.init_nodes(names=True)

    def simulate(self, steps=50, agent=None):
        # Menjalankan mesin komputasi numerik diskrit rekursif T(\Delta t) \to State Array 
        if agent is not None:
            # Memuat intervensi probabilitas model diskrit dari skrip komputasi AI Obyektif / Behavioral Markov
            from geevo.agent_simulation import AgentSimulator
            self.simulator = AgentSimulator(self.nodes, agent)
        else:
            self.simulator = Simulator(self.nodes)
        self.simulator.run(steps=steps)
        # Mengeluarkan output (Kumpulan vector waktu/historis hasil relasi transisi sistem persamaan)
        return self.simulator.monitoring

    def plot(self, figsize=(10, 4.5), save=False, filename="plots/graph.png", node_labels=None, edge_labels=None):
        # Penerjemahan Array Representatif Obyek Kelas menjadi Matriks Adjacency graf visual menggunakan pustaka matematika diskrit (NetworkX)
        g = nx.DiGraph() # Directed Graph: Graf berarah
        g.add_edges_from(sorted(self.edge_list))
        if self.plot_pos is None:
            try:
                # Algoritma Ploting Planar: Pemetaan fungsi isomorfik \mathcal{R}^2 (Titik planar kanvas agar Edge saling silang terminimalisir)
                pos = nx.planar_layout(g, scale=10)
            except nx.NetworkXException:
                # Spring / Fruchterman-Reingold layout solver: iterasi fisika mekanika tolakan node elastis
                print("Graph is probably not planar, so I choose spring layout for plotting.")
                pos = nx.spring_layout(g)
        else:
            pos = self.plot_pos
            
        # Merender diagram 2D Kartesian
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

        # Ekspor matriks array grafis raster memori 
        if save is True:
            plt.savefig(filename, dpi=300)
        plt.show()

    def get_nodes_of(self, node_type):
        """Mempartisi dan memfilter array N mengembalikan hanya kelas tipe titik observasi spesifik (Set Subset)."""
        return [n for n in self.nodes if isinstance(n, node_type)]
