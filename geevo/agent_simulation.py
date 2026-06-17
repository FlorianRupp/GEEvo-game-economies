import random
from geevo.nodes import Converter
from geevo.simulation import Simulator

class Agent:
    def __init__(self, behavior="random"):
        """
        Inisialisasi Profil Sikap Agen (Agent Behavior Profile).
        behavior: 'aggressive', 'passive', 'random' merepresentasikan fungsi stokastik (pengambilan keputusan acak)
        untuk mengendalikan jalannya transfer (transisi state) antar titik dalam graf.
        """
        self.behavior = behavior

    def play_step(self, graph, call_chain):
        # Array/Vektor berisi Converter Node yang membutuhkan pengawasan manuver agen (Kumpulan \mathcal{C})
        manual_converters = []
        for n in graph:
            # Memfilter node Converter (titik fungsi transformasi) di mana automasi diset false
            if isinstance(n, Converter) and getattr(n, 'is_auto', True) is False:
                # Eleminasi Converter yang diumpankan murni dari probabilitas Gate (RandomGate) -- 
                # yang secara topologi dikontrol probabilitas Markov dan tidak memiliki state/kolam
                if n.input_edges and type(n.input_edges[0].node).__name__ == "RandomGate":
                    continue
                # Menambahkan vertex yang lolos seleksi
                manual_converters.append(n)

        # Base case: Jika himpunan (set) titik intervensi manual adalah nol, maka kembali
        if not manual_converters:
            return

        # Logika Keputusan Aksi berdasarkan Parameter Skalar (Behavior):
        if self.behavior == "aggressive":
            # Perilaku Agresif (Greedy): Langsung picu (trigger) \forall C \in manual\_converters secara maksimum
            # Asal constraint \ge memenuhi batas bawah sumber daya yang dibutuhkan (terjadi di dalam node logic)
            for c in manual_converters:
                c.consume(call_chain, force_agent=True)
                
        elif self.behavior == "passive":
            # Perilaku Pasif (Synchronous Batch): Menunggu hingga seluruh sistem/skill state memiliki sumber daya lengkap.
            # Operasi Boole Logika AND / Himpunan Irisan terhadap semua prasyarat c \in manual_converters
            all_ready = True
            for c in manual_converters:
                if getattr(c, 'called', False):
                    # Flag jika fungsi relasi konversi f(x) sudah tereksekusi di sub-putaran waktu ini (Telah dipanggil)
                    all_ready = False
                    break
                    
                ready = True
                # Memverifikasi relasi inequalitas sumber daya dari koneksi masuk: 
                # Nilai pool m harus \ge batas transfer minimum value e
                # Skip node RandomGate karena tidak memiliki atribut 'pool'
                for input_e in c.input_edges:
                    if not hasattr(input_e.node, 'pool'):
                        continue
                    if not input_e.node.pool >= input_e.value:
                        ready = False
                        break
                if not ready:
                    all_ready = False
                    break
            
            # Jika semua proposisi Boolean bernilai T (True), eksekusi vektor konversi bersamaan (batch)
            if all_ready:
                for c in manual_converters:
                    c.consume(call_chain, force_agent=True)
                    
        elif self.behavior == "random":
            # Perilaku Probabilistik (Stochastic / Monte-Carlo based):
            # Pengambilan iterasi setiap c, dan dieksekusi dengan peluang sukses (success probability) p = 0.5 (Distribusi seragam)
            for c in manual_converters:
                if random.random() < 0.5:
                    c.consume(call_chain, force_agent=True)


class AgentSimulator(Simulator):
    """
    Ekstensi Kelas Induk (Inheritance dari Simulator) untuk mewadahi perhitungan graf 
    seiring siklus diskrit t (waktu diskrit iteratif), dengan kehadiran pengambilan intervensi stokastik agen.
    """
    def __init__(self, graph, agent):
        super().__init__(graph)
        # Menetapkan obyek Agent
        self.agent = agent

    def run(self, steps=10):
        # Iterasi langkah diskrit: Setiap transisi dari t ke t+1 (selama \Delta t)
        for _ in range(steps):
            # Inisiasi pasokan state awal ke dalam graf 
            # (Memasukkan boundary values dari node Source \mathcal{S})
            for source in self.sources:
                source.step([])
                
            # Intervensi aksi deterministik/stokastik ke dalam matriks nilai yang bergerak 
            self.agent.play_step(self.graph, [])
            
            # Perekaman histori sekuens state (untuk membentuk vektor pemantauan deret waktu)
            self.monitor()

            # Mereset flag relasi fungsi \forall c \in Conveters, menyetel state eksekusinya = False
            for c in self.converters:
                c.called = False
                
        # Setelah simulasi \Delta t iterasi selesai dijalankan (loop usai), reset nilai integral
        # di seluruh titik Pool supaya siap pada \text{run}() selanjutnya   
        [p.reset() for p in self.pools]
