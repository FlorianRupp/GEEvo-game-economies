"""
gdd_extractor.py
================
Mengekstrak konfigurasi ekonomi dari PDF Game Design Document (GDD)
dan membangkitkan graf ekonomi GEEvo secara otomatis.
"""

from geevo import nodes as n
from geevo.graph import Graph
from geevo.evolution.generator import EvolutionaryGraphGeneration


# Kata kunci per tipe node — diurutkan dari yang paling spesifik ke umum
# agar keyword overlap antar kategori minimal.
_KEYWORD_MAP = {
    n.Source: [
        'drop', 'reward', 'quest', 'loot', 'gather', 'generate',
        'login', 'spawn', 'earn', 'income', 'produce', 'grant',
    ],
    n.Pool: [
        'gold', 'currency', 'resource', 'inventory', 'wealth',
        'wallet', 'balance', 'stock', 'storage', 'mana', 'xp',
        'experience', 'health', 'stamina', 'energy',
    ],
    n.Converter: [
        'shop', 'crafting', 'craft', 'trade', 'market', 'exchange',
        'upgrade', 'buy', 'sell', 'convert', 'forge', 'merchant',
        'vendor', 'purchase', 'recipe',
    ],
    n.RandomGate: [
        'chance', 'luck', 'rng', 'random', 'gamble', 'probability',
        'gacha', 'dice', 'jackpot', 'lottery', 'drop rate', 'loot table',
    ],
}

_FALLBACK_CONF = {n.Source: 2, n.Pool: 4, n.Converter: 2, n.RandomGate: 1}


def _read_pdf_pages(pdf_path, max_pages=50):
    """Membaca teks PDF, maks max_pages halaman. Prioritas fitz, fallback PyPDF2."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = [doc[i].get_text().lower() for i in range(min(max_pages, len(doc)))]
        doc.close()
        return pages
    except ImportError:
        pass

    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return [
                (reader.pages[i].extract_text() or '').lower()
                for i in range(min(max_pages, len(reader.pages)))
            ]
    except ImportError:
        raise ImportError(
            'Tidak ada library PDF tersedia.\n'
            'Install salah satu: pip install PyMuPDF  atau  pip install PyPDF2'
        )


def _count_keywords(pages):
    """Menghitung total frekuensi kemunculan keyword per tipe node dari semua halaman."""
    weights = {node_type: 0 for node_type in _KEYWORD_MAP}
    for page_text in pages:
        for node_type, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                weights[node_type] += page_text.count(kw)
    return weights


def _weights_to_conf(weights, max_nodes):
    """
    Mengonversi bobot frekuensi ke distribusi jumlah node secara proporsional.
    Setiap tipe node dijamin minimal 1 node.
    Mengembalikan None jika total bobot = 0 (tidak ada keyword ditemukan).
    """
    total = sum(weights.values())
    if total == 0:
        return None
    return {
        node_type: max(1, round((w / total) * max_nodes))
        for node_type, w in weights.items()
    }


class GDDGraphExtractor:
    """
    Pipeline lengkap dari PDF Game Design Document ke Graph ekonomi GEEvo.

    Langkah internal:
      1. Baca teks PDF (fitz / PyPDF2 sebagai fallback)
      2. Hitung frekuensi keyword per tipe node
      3. Normalisasi ke distribusi conf {NodeType: count}
      4. Jalankan EvolutionaryGraphGeneration untuk membangkitkan topologi
      5. Kembalikan obyek Graph siap pakai

    Contoh penggunaan:
        extractor = GDDGraphExtractor('Gdd/MMORPG_GameDesign_v27.pdf')
        graph = extractor.extract()
        graph.simulate(10)
        graph.plot()
    """

    def __init__(self, pdf_path, max_nodes=13, max_pdf_pages=50):
        self.pdf_path = pdf_path
        self.max_nodes = max_nodes
        self.max_pdf_pages = max_pdf_pages
        self.conf = None
        self.graph = None
        self._keyword_weights = None

    def extract_conf(self):
        """
        Membaca PDF dan mengekstrak distribusi node (conf dict).

        Returns
        -------
        dict
            {NodeType: count} kompatibel dengan EvolutionaryGraphGeneration.
        """
        pages = _read_pdf_pages(self.pdf_path, self.max_pdf_pages)
        self._keyword_weights = _count_keywords(pages)
        self.conf = _weights_to_conf(self._keyword_weights, self.max_nodes)

        if self.conf is None:
            print(f'[GDDGraphExtractor] Peringatan: tidak ada keyword ditemukan di {self.pdf_path!r}.')
            print('[GDDGraphExtractor] Menggunakan konfigurasi fallback.')
            self.conf = _FALLBACK_CONF.copy()

        self._print_summary()
        return self.conf

    def generate_graph(self, egg_steps=150_000):
        """
        Membangkitkan graf ekonomi menggunakan EvolutionaryGraphGeneration.
        Harus dipanggil setelah extract_conf(), atau gunakan extract() sekaligus.

        Parameters
        ----------
        egg_steps : int
            Batas iterasi maksimum EGG. Default 150_000.

        Returns
        -------
        Graph
        """
        if self.conf is None:
            raise RuntimeError('Panggil extract_conf() terlebih dahulu, atau gunakan extract().')

        egg = EvolutionaryGraphGeneration(self.conf)
        egg.run(egg_steps)
        self.graph = Graph(self.conf, egg.get_best())
        return self.graph

    def extract(self, egg_steps=150_000):
        """
        Pipeline lengkap: ekstrak conf dari PDF lalu bangkitkan graf.

        Parameters
        ----------
        egg_steps : int
            Batas iterasi EGG. Default 150_000.

        Returns
        -------
        Graph
            Graf ekonomi siap disimulasikan.
        """
        self.extract_conf()
        return self.generate_graph(egg_steps)

    def _print_summary(self):
        if self._keyword_weights is None:
            return
        print(f'\n[GDDGraphExtractor] Sumber: {self.pdf_path}')
        print('Frekuensi keyword per tipe node:')
        for node_type, w in self._keyword_weights.items():
            print(f'  {node_type.__name__:>12s} : {w}')
        print('Distribusi node (conf):')
        for node_type, count in self.conf.items():
            print(f'  {node_type.__name__:>12s} : {count}')


# Fungsi kompatibilitas — memanggil GDDGraphExtractor.extract_conf() secara langsung.
def extract_conf_from_gdd(pdf_path, max_nodes=13):
    """Wrapper kompatibilitas. Gunakan GDDGraphExtractor untuk pipeline lengkap."""
    return GDDGraphExtractor(pdf_path, max_nodes=max_nodes).extract_conf()
