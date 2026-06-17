import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from geevo.nodes import Pool, Source, Converter


class Simulator:
    """
    Model Matematika Evaluator Graf: Mesin eksekutor deret iterasi komputasi yang 
    mensimulasikan evolusi aliran graf (Graph Flow) berbasis waktu diskret (Discrete Time Simulation).
    """
    def __init__(self, graph):
        self.graph = graph
        # Dictionary untuk mencatat vektor seri deret waktu \vec{X}(t) dari jumlah elemen / nilai integral yang
        # berada pada \forall P \in Pool di graf (Titik State / Node Penyimpanan State).
        self.monitoring = {p: [] for p in self.graph if isinstance(p, Pool)}
        
        # Mengekstrak himpunan bagian titik pada graf V, menjadi himpunan node Pool (Simpanan), Source (Sumber Pasokan), dan Converter (Transformasi)
        self.pools = [n for n in self.graph if isinstance(n, Pool)]
        self.sources = [n for n in self.graph if isinstance(n, Source)]
        self.converters = [n for n in self.graph if isinstance(n, Converter)]

    def run(self, steps=10):
        # Perulangan untuk pergeseran dimensi t = 0 hingga t = steps-1. (Sistem Persamaan Beda / Differensi Orde Pertama)
        for _ in range(steps):
            # Iterasi memasukkan variabel bebas batas (boundary parameters) f_S(x) \forall Source \in S
            for source in self.sources:
                source.step([])
                
            # Menggandakan titik rekaman sekuens waktu t untuk membentuk profil distribusi historis \vec{h}(t)
            self.monitor()

            # Proses reset state flag evaluasi fungsi transformasi: Memberitahu graf untuk "membuka kembali gerbang" e \in Edge 
            # dan meniadakan parameter perantara / intermediary untuk menyambut t+1.
            for c in self.converters:
                c.called = False
            # for p in self.pools:
            #     p.called = 0
            
        # Perlakuan inisialisasi / kliring post-simulasi: 
        # State Value (\forall p) ditetapkan = 0 kembali pasca integral keseluruhan dijalankan
        [p.reset() for p in self.pools]

    def monitor(self):
        # Merekam / Snapshot metrik P(t) \forall node, dicatat ke skalar array historis (monitoring[node])
        for node in self.graph:
            if isinstance(node, Pool):
                self.monitoring[node].append(node.pool)

    def plot_monitor(self, drains=True, figsize=(10, 7), labels=None, save=False, filename="plots/graph.png",
                     xticks=None):
        """
        Penyajian fungsi visual geometri analitik kartesian dimensi 2 (Grafik fungsi f(x) = y \rightarrow t \mapsto Pool_Amt)
        menggunakan library matplotlib.
        """
        # Konfigurasi figur proyeksi dan batas ruang (axis) kanvas plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        count = 0
        
        # Plotting himpunan nilai integral \vec{X}(t) terhadap axis x (Rentang Waktu Diskrit)
        for k, v in self.monitoring.items():
            if drains is True:
                if labels is None:
                    ax.plot(list(range(len(v))), v, label=k.name)
                else:
                    # Menerapkan shift skala indeks axis x sebesar +1 (dimulai dari 1 ke max(len(v)))
                    ax.plot(list(range(1, len(v) + 1)), v, label=labels[count])
                    count += 1
            else:
                if "drain" not in k.name.lower():
                    ax.plot(list(range(len(v))), v, label=k.name)
                    
        # Menamai vektor absis x dan y pada fungsi koordinat analitik
        ax.set_xlabel("Time steps", fontsize=14)
        ax.set_ylabel("Amount", fontsize=14)
        
        # Perlakuan khusus pengkalakan integer untuk sumbu independen (X): Interpolasi visual 1 step interval
        if xticks is not None:
            plt.xticks([1, 2, 3, 4, 5], fontsize=12)
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        else:
            plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.ylim(0, 17) # Opsional: Pembatasan supremum / batas domain Y (asymptote horizontal bayangan)
        plt.legend()
        fig.patch.set_facecolor('white') # Set representasi RGB/matriks background visual ke #FFFFFF
        plt.tight_layout() # Algoritma pemadatan batas kanvas auto
        
        # Export proyeksi bidang datar (bidang komposit grid) ke data binary raster resolusi 300 Dot per Inci (dpi)
        if save is True:
            plt.savefig(filename, dpi=300)
        plt.plot()

        if labels is not None:
            # Mencetak elemen limit fungsi ujung: \lim_{t \to steps} \vec{v}(t) = \vec{v}[-1]
            print(", ".join([f"{l}: {v[-1]}" for l, v in zip(labels, self.monitoring.values())]))
