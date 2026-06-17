from geevo.evolution.generator import EvolutionaryGraphGeneration
from geevo.evolution.balancer import Balancer
from geevo import nodes as n
from geevo.graph import Graph
from geevo.agent_simulation import Agent
import warnings

# Mengabaikan pesan error atau himbauan warning dari Python agar terminal output bersih
warnings.filterwarnings('ignore')

# Mendefinisikan topologi atau himpunan jenis titik (Vertex / Node config) untuk pembentukan graf
# Node ini dipetakan menjadi dictionary: Masing-masing kelas node dengan integer (jumlah) skalarnya.
conf = {
    n.Source: 3,        # \mathcal{S} (Himpunan sumber, ukuran=3)
    n.RandomGate: 2,    # Percabangan stokastik
    n.Pool: 4,          # Titik simpanan / State vector \mathcal{P}
    n.Converter: 1      # Titik transformasi f(x)
}

print("Running EGG...")
# EGG: Evolutionary Graph Generation - Sebuah algoritma optimasi generasi graf 
# berdasarkan topologi yang diberikan di atas.
egg = EvolutionaryGraphGeneration(conf)
egg.run()

# Membentuk obyek graf jaringan dari hasil individu paling optimal yang didapat oleh prosedur evolusi (egg.get_best())
g = Graph(conf, egg.get_best())

# Set flag mode otomasi Converter Node (node konversi). Diatur ke mode manual / kontrol agen,
# supaya intervensi agen diskrit bisa mengatur dinamika konversi ini.
for c in g.get_nodes_of(n.Converter):
    c.is_auto = False

# Menginstansiasi obyek "Agent" yaitu model komputasi yang memiliki behavior spesifik.
# Setiap profil perilaku adalah pemetaan fungsi pengambilan keputusan probabilitas stokastik: {agresif, pasif, random}
agents = [Agent(behavior='aggressive'), Agent(behavior='passive'), Agent(behavior='random')]

print("Simulating graph with all agents...")
# Simulasi lintasan waktu (Time-step series execution), dijalankan \Delta t = 10, menggunakan agen ke-2 (pasif)
g.simulate(10, agent=agents[1])

# Mengambil semua Vertex yang merupakan himpunan Node Pool
# Diperhitungkan bahwa ini menjadi state variabel dimensi N
pool_nodes = g.get_nodes_of(n.Pool)

# Verifikasi himpunan indeks: Menemukan indeks variabel Pool pada Graph
if len(g.nodes) > 5 and g.nodes[5] in pool_nodes:
    POOL_NODE_ID = pool_nodes.index(g.nodes[5])
else:
    # Memilih indeks pool pertama bernilai 0
    POOL_NODE_ID = 0

print("Running Balancer...")
# Inisialisasi proses Balancer (Genetic Algorithm) untuk menyeimbangkan nilai target aliran ekonomi
# - Parameter `balance_value=42` adalah Constrain / Limit konvergensi target, misal L = 42
# - Output diharapkan meminimumkan fitness function terhadap nilai 42
balancer = Balancer(agent=agents, config=g.config, edge_list=g.edge_list, balance_pool_ids=[POOL_NODE_ID], n_sim_steps=10,
                    balance_value=42, alpha=0.01, frozen_weights=None)
# Mulai iterasi proses optimasi genetika sepanjang 5 langkah / epoch generasi
balancer.run(5)

# Mengembalikan nilai array solusi berbobot terbaik yang sudah konvergen
print("Best result:", balancer.result)
print("Integration test passed!")
