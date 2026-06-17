import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Edge:
    """
    Representasi relasional himpunan sisi (Directed Edge) E(u, v) pada Graf Berarah (DiGraph) G = (V, E).
    Sisi meneruskan vektor besaran ke titik simpul tujuan.
    """
    def __init__(self, node, value, name=None, node_id=None):
        # Skalar (float/integer) yang mendefinisikan bobot relasi atau limitasi kecepatan
        # Untuk cabang probabilitas, value \in [0, 1] bertindak layaknya peluang Markov (Transition Probability)
        self.value = value
        # Target node v
        self.node = node
        if name is not None:
            self.node.name = name
        self.node_id = node_id


class Node(ABC):
    """
    Abstraksi Basis (Polimorfisme) untuk Obyek Simpul V \in Graf.
    Memiliki sifat In-Degree (Koneksi Masuk) dan Out-Degree (Koneksi Keluar).
    """
    ALLOWED_INPUT = []   # Konstrain Domain fungsi input
    ALLOWED_OUTPUT = []  # Konstrain Kodomain fungsi output
    EMPTY_INPUT = True
    EMPTY_OUTPUT = True
    MAX_INPUT = 0        # Rentang skalar maksimum kardinalitas In-Degree \leq k
    MAX_OUTPUT = 0       # Rentang skalar maksimum kardinalitas Out-Degree \leq c
    COLOR = None

    def __init__(self, name=None, id=None):
        self.name = name
        # Koleksi Vektor Adjacency List untuk menyimpan lintasan masuk dan lintasan keluar
        self.input_edges = []
        self.output_edges = []
        self.id = id

    def check(self, node):
        """
        Pengecekan Hukum Transitivitas Relasi dan Konstrain Topologi Jaringan.
        """
        if type(self) not in node.ALLOWED_INPUT or type(node) not in self.ALLOWED_OUTPUT:
            raise ValueError(f"Connections of {type(self)} and {type(node)} are not allowed.")
        node_edges = [e.node for e in node.input_edges]
        node_edges.extend([e.node for e in node.output_edges])
        # Aksioma Ireflexive/Acyclic: Tidak membolehkan pembentukan loop tertutup terhadap simpul yang sama yang telah terhubung sebelumnya
        if self in node_edges:
            raise ValueError(f"Nodes are already connected.")
        # Pengecekan limitasi maksimum himpunan titik ketetanggaan masuk N^-(v)
        if node.MAX_INPUT == len(node.input_edges):
            raise ValueError(f"{type(node).__name__} has already the maximum of {node.MAX_INPUT} inputs.")
        # Pengecekan limitasi maksimum himpunan titik ketetanggaan keluar N^+(v)
        if self.MAX_OUTPUT == len(self.output_edges):
            raise ValueError(f"{type(self).__name__} has already the maximum of {self.MAX_OUTPUT} outputs.")

    def check_connection(func):
        """Dekorator (Higher-Order Function) untuk verifikasi prasyarat alokasi fungsi graf."""
        def wrapper(self, node, value, name=None, node_id=None):
            self.check(node)
            func(self, node, value, name, node_id)

        return wrapper

    @staticmethod
    def init_or_random(value, a=0, b=2, integer=True):
        # Pemodelan fungsi acak (Stochastic Sampling) menggunakan seragam riil (0 ~ 1) jika integer bernilai False,
        # dan seragam pseudo-random integer [a, b] jika True.
        if value is None:
            if integer is True:
                return random.randint(a, b)
            else:
                return random.random()
        else:
            return value

    @check_connection
    def connect(self, node, value, name=None, node_id=None):
        """Operasi Pembentukan Sisi (Aljabar Penambahan Ruang Graph Edge). A \rightarrow B"""
        self.output_edges.append(Edge(node, value, name, node_id=node_id))
        node.input_edges.append(Edge(self, value, name))

    def disconnect(self, node):
        """Operasi Pengurangan Himpunan Sisi Graf. A \& B diskonektif."""
        assert node in self.get_output_nodes(), "Nodes are not connected"
        self.output_edges.remove([e for e in self.output_edges if e.node is node][0])
        node.input_edges.remove([e for e in node.input_edges if e.node is self][0])

    def get_output_nodes(self):
        return [edge.node for edge in self.output_edges]

    def get_input_nodes(self):
        return [edge.node for edge in self.input_edges]

    def step(self, call_chain):
        # Eksekusi unit transformasi/iteratif (Differensi Langkah Waktu) suatu node.
        # Menghasilkan state baru N = {t+1}
        pass

    def __str__(self):
        input_edges = [type(e.node).__name__ for e in self.input_edges]
        output_edges = [type(e.node).__name__ for e in self.output_edges]
        return f"{type(self).__name__}: [input: {input_edges}, output: {output_edges}]"

    def update_edge_value(self, node, value):
        # Operasi skalar substitusi matriks/relasi pembobot sisi: e_k := value baru
        [e for e in self.output_edges if e.node == node][0].value = value
        [e for e in node.input_edges if e.node == self][0].value = value

    def get_state(self):
        """Mengukur Penalty Score metrik kelayakan titik sebagai elemen fitness function genetik."""
        score = 0
        if len(self.input_edges) == 0 and self.EMPTY_INPUT is False:
            score -= 1
        if len(self.output_edges) == 0 and self.EMPTY_OUTPUT is False:
            score -= 1
        return score


class Source(Node):
    """
    Generator (S_0): Batas Titik Awal. Mensuplai skalar sumber daya tanpa batas
    ke sistem (In-Degree = 0).
    """
    EMPTY_INPUT = True
    EMPTY_OUTPUT = False
    MAX_INPUT = 0
    MAX_OUTPUT = 3
    COLOR = "limegreen"

    def step(self, call_chain):
        for edge in self.output_edges:
            # Transformasi nilai x_out dieksekusi maju ke depan ke subgraf selanjutnya (\Delta iterasi integral ke nodes)
            edge.node.consume(self.drop_to(edge), call_chain)
        for edge in self.output_edges:
            edge.node.step(call_chain)

    def drop_to(self, edge):
        # Mengembalikan koefisien tetap limit kecepatan laju output (x(t) output)
        return edge.value


class Pool(Node):
    """
    State Variabel / Variabel Basis (\mathcal{P}):
    Simpul penyimpanan variabel riil yang menumpuk / menjumlah sumber daya (resource).
    \Delta P = P(t_{n+1}) = P(t_n) + x_{in} - x_{out}
    """
    EMPTY_INPUT = False
    EMPTY_OUTPUT = True
    MAX_INPUT = 2
    MAX_OUTPUT = 3
    COLOR = "skyblue"

    def __init__(self, name=None, id=None):
        super().__init__(name, id=id)
        # Besaran absolut (Integral akumulatif)
        self.pool = 0

    def step(self, call_chain):
        # Deteksi Siklus Tertutup (Loop Detection): 
        # Untuk mencegah infinite recursion \to StackOverflow. Merupakan algoritma pendeteksi DAG (Directed Acyclic Graph).
        if self in call_chain:
            return
        call_chain.append(self) # push ke array visit trail
        # Forward pass propagasi state
        for edge in self.output_edges:
            edge.node.consume(call_chain)

    def consume(self, value, call_chain):
        # Operasi pertambahan (Aljabar Penjumlahan Nilai State): S(t) + \Delta
        self.pool += value
        self.step(call_chain)

    def reset(self):
        self.pool = 0


class FixedPool(Pool):
    """
    Model Pool / State dengan Batasan Kapasitas Konstanta (\mathcal{C}_{max}).
    Variabel non-linier terikat f(x) = \min(\text{pool} + x, \text{fix}).
    """
    def __init__(self, name=None):
        super().__init__(name)

    def get_fix(self):
        # Mengekstrak supremum batasan atas / kapasitansi terbesar dari array laju bobot output
        return max([e.value for e in self.output_edges])

    def consume(self, value, call_chain):
        # Relasi inkualitas kondisional P < \max{E(o)}
        if not self.pool >= self.get_fix():
            self.pool += value
        self.step(call_chain)


class Converter(Node):
    """
    Fungsi Transformasi / Matriks Konversi Deterministik ($C$).
    Mengubah / menghapus elemen input menjadi kombinasi linier elemen output baru.
    """
    EMPTY_INPUT = False
    EMPTY_OUTPUT = False
    MAX_INPUT = 3
    MAX_OUTPUT = 1
    COLOR = "gold"

    def __init__(self, name=None, id=None, is_auto=True):
        super().__init__(name, id=id)
        self.called = False
        self.is_auto = is_auto

    def consume(self, call_chain, force_agent=False):
        # Hanya mengevaluasi relasi fungsi satu kali per detak t
        if self.called is False:
            # Seleksi mode aktivasi fungsi C(x) 
            if not self.is_auto and not force_agent:
                self.step(call_chain)
                return
            
            resources_available = True

            # Mekanisme bypass pengecualian topologi node probabilitas stokastik Markov (RandomGate):
            # Jika ANY input edge berasal dari RandomGate, langsung propagasi ke output dan return.
            if any(isinstance(e.node, RandomGate) for e in self.input_edges):
                self.output_edges[0].node.consume(self.output_edges[0].value, call_chain)
                self.called = True
                return

            # Mengevaluasi Operator Himpunan Boolean AND: 
            # Semua (forall) prasyarat sisi harus memenuhi x \geq limit konstraksi e
            # Hanya node yang memiliki atribut 'pool' (Pool/FixedPool) yang dievaluasi
            for input_e in self.input_edges:
                if not hasattr(input_e.node, 'pool') or not input_e.node.pool >= input_e.value:
                    resources_available = False
                    
            if resources_available is True:
                # Modifikasi (Variabel Operasi Pengurangan State Himpunan)
                for input_e in self.input_edges:
                    input_e.node.pool -= input_e.value
                # Propagasi pemetaan operan maju dari output f(x) \to O
                for output_e in self.output_edges:
                    output_e.node.consume(output_e.value, call_chain)
                self.called = True
            # Lanjutkan evaluasi langkah (Forward execution DAG)
            self.step(call_chain)


class RandomGate(Node):
    r"""
    Percabangan Aliran Terbobot (Weighted Stochastic Split / Rantai Markov).
    Mendistribusikan aliran sesuai koefisien proporsi nilai probabilitas $P \in [0, 1]$.
    """
    EMPTY_INPUT = False
    EMPTY_OUTPUT = False
    MAX_INPUT = 1
    MAX_OUTPUT = 3
    COLOR = "red"

    def step(self, call_chain):
        for edge in self.output_edges:
            edge.node.step(call_chain)

    def consume(self, entity, call_chain):
        # Array vektor probabilitas kejadian: \vec{p}
        probs = [edge.value for edge in self.output_edges]
        if int(entity) > 0:
            # Percobaan probabilitas majemuk distribusi multinomial dari library NumPy.
            # Mengukur vektor sampel k-observasi dari trial n eksperimen:
            # P(X = x) = \frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}
            counts = np.random.multinomial(int(entity), probs)
            
            # Mendistribusikan kembali bilangan cacah empiris (hasil observasi probabilistik multinomial) ke titik output
            for count, edge in zip(counts, self.output_edges):
                for _ in range(count):
                    # Evaluasi fungsi konversi jika edge.node adalah ruang transformasi
                    if isinstance(edge.node, Converter):
                        edge.node.consume(call_chain)
                    else:
                        edge.node.consume(1, call_chain)
        self.step(call_chain)


class Drain(Pool):
    r"""
    Batas Ujung Bawah (\text{Sink Node} / Pengeluaran Tak Terhingga).
    Node di mana limit dari t \to \infty graf akan membuang (sinkronisasi absorpsi material).
    Node ini memiliki karakteristik konstan positif untuk input dan konstan nol untuk set keluaran (MAX_OUTPUT = 0).
    """
    MAX_OUTPUT = 0
    COLOR = "darkorange"

    def consume(self, value=None, call_chain=None):
        for input_e in self.input_edges:
            # Algoritma konsumsi absolut: jika titik supply awal cukup, maka transfer elemen x dilakukan
            # (Memodelkan pembuangan variabel dalam sistem persamaan differensial graf aliran waktu).
            if input_e.node.pool >= input_e.value:
                input_e.node.pool -= input_e.value
                self.pool += input_e.value


class Result(Pool):
    def step(self):
        # Absorving point (Node terminal non-proses) -- Tidak memiliki forward pass
        pass


# Operasi Aljabar Konstrain (Boundary Conditions) Pembentukan Vektor Koneksi (DAG Topology Limit):
# Menentukan kodomain fungsi f(x) pada ruang state dan probabilitas koneksi spesifik matriks topologi Graf
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
