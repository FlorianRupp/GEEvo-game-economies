    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% File `TA-ITS.cls' ini ditujukan untuk menulis laporan
%%% Tugas Akhir (Skripsi) menggunakan LaTeX untuk program sarjana
%%% di Institut Teknologi Sepuluh Nopember, Surabaya.
%%%
%%% Komentar, saran, koreksi, modifikasi untuk file ini dipersilakan
%%%
%%% Dibuat oleh: Dr. Mahmud Yunus, M.Si
%%% NIP.19620407 198703 1 005
%%% <yunusm@matematika.its.ac.id>
%%%
%%% Revisi oleh: Mohamad Ilham Dwi Firmansyah
%%% Matematika ITS 2016
%%% <syah3946@gmail.com>
%%% Revisi: Format A5
%%%
%%% Revisi oleh: Alvian Alif Hidayatullah
%%% Matematika ITS 2018 
%%% <alvinelbarca08@gmail.com>
%%% Revisi: Format A4 (2022)
%%%
%%% Revisi oleh: Ridho Nur Rohman Wijaya
%%% Matematika ITS 2018
%%% <ridhonurrohmanwijaya@gmail.com>
%%% Revisi: Generalisasi dan penyempurnaan Format A4 (2022)
%%%
%%% Revisi terakhir oleh : Nathaniel Takeshi Inatori Lauw
%%% Matematika ITS 2021
%%% <nathanieltakeshi9320@gmail.com>
%%% Revisi: Penyempurnaan format agar lebih rapi; peningkatan keterbacaan kode [agar lebih mudah dipahami] (2024)
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CTRL + / UNTUK KOMENTAR (COMMENTING)

\ProvidesClass{TA-ITS}
\NeedsTeXFormat{LaTeX2e}

% PAKAI INI UNTUK PRINT DRAFT & PROPOSAL
% \LoadClass[12pt, oneside, openright, final]{book}


% PAKAI INI UNTUK PRINT BUKU
\LoadClass[12pt, twoside, openright]{book}

\PassOptionsToPackage{hyphens}{url} % Biar URL bisa pecah

\RequirePackage{geometry} % Margin
\RequirePackage{amsmath, amsthm, amsfonts, amssymb} % Klasik buat matematika
\RequirePackage[inline]{enumitem} % Kustomisasi penomoran
\RequirePackage{fancyhdr} % Header & Footer
\RequirePackage{hhline, colortbl} % Ketebalan tabel gg
\RequirePackage{xcolor} % Warna lebih banyak
\RequirePackage{graphicx} % Masukkin gambar
\RequirePackage{indentfirst} % Indentasi paragraf pertama
\RequirePackage[explicit]{titlesec} % Kustomisasi bab (chapters) dan bagian (sections)
\RequirePackage{multicol, multirow} % Kolom dan baris ganda; kolom ganda untuk teks, gabung baris untuk tabel -> multirow
\RequirePackage{titletoc} % Kustomisasi tampilan daftar isi
\RequirePackage{appendix} % Lampiran
\RequirePackage{xurl} % Buat link URL
\RequirePackage{hyperref} % Pengaturan pranala
\RequirePackage{pdfpages} % Sisip file PDF
\RequirePackage{float} % Float; untuk menyisipkan gambar lebih rapi
\RequirePackage{placeins} % Pendukung package `float`
\RequirePackage{tikz, pgfplots} % Untuk sketsa grafis
\RequirePackage{etoolbox} % Peletakkan elemen
\RequirePackage{setspace} % Spasi teks
\RequirePackage{wrapfig} % Untuk pengemasan gambar pada teks
\RequirePackage{tabularx} % Pembuatan tabel dengan lebar adaptif
\RequirePackage{calc} % Kalkulasi penghitungan panjang
\RequirePackage{subcaption} % Untuk anakan gambar
\RequirePackage[normalem]{ulem} % Pengaturan underline
\RequirePackage{tabularx} % Tabulasi selebar kertas
\RequirePackage{makecell} % Bikin sel tabel khusus
\RequirePackage{etoolbox} % sc: trust me, bro
\RequirePackage{emptypage} % Untuk halaman tanpa nomor
\RequirePackage{ifthen} % Seleksi pilihan
\RequirePackage{iftex} % Pilih engine
\RequirePackage{braket} % Notasi Dirac

\RequirePackage{algorithm} % Penulisan algoritma
\RequirePackage{algpseudocode} % Penulisan algoritma; lebih spesifik

% Daftar Pustaka
% Version 1
% \RequirePackage[backend=biber,style=numeric,sorting=none]{biblatex}
% \addbibresource{assets/pustaka.bib}

% Version 2
% \RequirePackage[nosectionbib, notocbib]{apacite}

% \RequirePackage[english]{babel}

% Version 3
\RequirePackage[nosectionbib, notocbib]{apacite}
\AtBeginDocument{%
    \renewcommand{\BBAA}{ dan }   % Dua penulis
    \renewcommand{\BBAB}{ dan }   % Sebelum penulis terakhir
    \renewcommand{\BOthers}{dkk} % Lebih dari dua penulis
}

\RequirePackage[english]{babel}
% =================================================
% THEOREMS & DEFINITIONS
% Untuk pendefinisian teorema, definisi dan lainnya
% =================================================

\theoremstyle{definition}

% Teorema
\newtheorem{theorem}{Theorem}[section]
\newtheorem{teorema}{Teorema}[section]

% Definisi
\newtheorem{definition}[theorem]{Definition}
\newtheorem{definisi}[theorem]{Definisi}

% Contoh
\newtheorem{example}[theorem]{Example}
\newtheorem{contoh}[teorema]{Contoh}

% ====================================
% OPTIONS
% Untuk pilah pilih pengaturan package
% ====================================

\newif\iftatimes\tatimesfalse
\newif\iftabera\taberafalse
\newif\ifhvs\hvstrue

\DeclareOption{times}{\tatimestrue}
\DeclareOption{bera}{\taberatrue}
\DeclareOption{a5}{\hvsfalse}
\ProcessOptions\relax

\usepackage{tikz}
\usetikzlibrary{shapes.multipart, positioning, arrows.meta, calc}

\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{natbib}
\usepackage{pifont}
\usepackage{soul}

\usepackage[utf8]{inputenc}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\usepackage{algorithm}
\usepackage{algpseudocode}

% Di preamble:
\algnewcommand\algorithmicinput{\textbf{Input:}}
\algnewcommand\algorithmicoutput{\textbf{Output:}}
\algnewcommand\INPUT{\item[\algorithmicinput]}
\algnewcommand\OUTPUT{\item[\algorithmicoutput]}

% Font pakai Times New Roman
\ifXeTeX
    \usepackage{fontspec}
    \iftatimes
        \setmainfont{Times New Roman}
    \fi

\else
    \iftatimes
        \usepackage{mathptmx}
    \fi
\fi

% Font pakai Bera
\iftabera
    \RequirePackage{bera}
\fi

% Ukuran kertas
\ifhvs
    % Margin + header settings
    \geometry{
        a4paper,
        top = 3cm,
        bottom = 2.5cm,
        left = 3cm,
        right = 2cm,
    }
\else
    % Margin + header settings
    \geometry{
        a5paper,
        top = 3cm,
        bottom = 2.5cm,
        left = 3cm,
        right = 2cm,
    }
\fi

% =============================
% LAYOUT SETTINGS
% Pengaturan tata letak dokumen
% =============================

% Pengaturan pranala (hyperlink)
\hypersetup{
    colorlinks = true,
    allcolors = black,
}

% Gaya halaman
\fancypagestyle{draf}{
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyhead[L]{}
    \fancyhead[R]{}
    \fancyfoot[L]{}
    \fancyfoot[C]{}
    \fancyfoot[R]{\thepage}
}

% Gaya halaman (TA)
\fancypagestyle{TA}{
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyhead[L]{}
    \fancyhead[R]{}
    \fancyfoot[C]{}
    \fancyfoot[LE, RO]{\thepage}
}

% \fancypagestyle{TA}{
%     \fancyhf{}
%     \renewcommand{\headrulewidth}{0pt}
%     \renewcommand{\footrulewidth}{0pt}
%     \fancyfoot[LE]{\thepage}
%     \fancyfoot[RO]{\thepage}
% }


\setstretch{1} % Spasi antar baris

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PERINTAH KHUSUS
% Dibuat khusus untuk kelas ini; dapat diubah nilainya untuk variabel pengguna
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ++++++++++++++++++++++++++++++++++++++++++++
% JUDUL TA
% Perintah untuk memperbarui judul Tugas Akhir
% ++++++++++++++++++++++++++++++++++++++++++++

% Contoh penjelasan penggunaan
\newcommand\@judul{Judul T.A.} % Judul Tugas Akhir dalam Bahasa Indonesia; Perintah ini untuk mendeklarasikan suatu perintah dalaman di dalam kelas
\DeclareRobustCommand{\Judul}[1]{ % Deklarasi perintah yang bisa digunakan
    \renewcommand{\@judul}{#1} % Perbarui data di dalam perintahnya
}
\newcommand\@title@{The Title} % Judul Tugas Akhir dalam Bahasa Inggris
\DeclareRobustCommand{\Title}[1]{
    \renewcommand{\@title@}{#1}
}

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% KETERANGAN MAHASISWA
% Perintah untuk memperbarui data mahasiswa YANG MENGERJAKAN TUGAS AKHIR
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\newcommand\@nama{Nama Mahasiswa} % Nama mahasiswa
\DeclareRobustCommand{\Nama}[1]{
    \renewcommand{\@nama}{#1}
}
\newcommand\nama@kecil[1]{Nama Kecil} % Nama alias atau nama kecil
\DeclareRobustCommand{\NamaKecil}[1]{
    \renewcommand{\nama@kecil}{}
}
\newcommand\@NRP{1122334455} % Nomor Registrasi Pokok (NRP) mahasiswa
\DeclareRobustCommand{\NRP}[1]{
    \renewcommand{\@NRP}{#1}
}

\newcommand\kode@TA{XX112233} % Kode mata kuliah untuk Tugas Akhir
\DeclareRobustCommand{\KodeTA}[1]{
    \renewcommand{\kode@TA}{#1}
}

\newcommand\@departemen{Matematika} % Asal departemen
\DeclareRobustCommand{\Departemen}[1]{
    \renewcommand{\@departemen}{#1}
}
\newcommand\@department{Mathematics} % Asal departemen dalam Bahasa Inggris
\DeclareRobustCommand{\Department}[1]{
    \renewcommand{\@department}{#1}
}
\newcommand\@laboratorium{Pembelajaran Mesin dan Big Data} % Asal laboratorium atau peminatan atau bidang studi
\DeclareRobustCommand{\Laboratorium}[1]{
    \renewcommand{\@laboratorium}{#1}
}
\newcommand\@fakultas{Sains dan Analitika Data} % Asal fakultas
\DeclareRobustCommand{\Fakultas}[1]{
    \renewcommand{\@fakultas}{#1}
}
\newcommand\@faculty{Science and Data Analytics} % Asal fakultas 
\DeclareRobustCommand{\Faculty}[1]{
    \renewcommand{\@faculty}{#1}
}
\newcommand\singkatan@fakultas{FSAD} % Singkatan fakultas
\DeclareRobustCommand{\SingkatanFakultas}[1]{
    \renewcommand{\singkatan@fakultas}{#1}
}
\newcommand\faculty@abbreviation{SCIENTICS} % Singkatan fakultas dalam Bahasa Inggris
\DeclareRobustCommand{\SingkatanFakultasEng}[1]{
    \renewcommand{\faculty@abbreviation}{#1}
}

\newcommand\@tanggal{1} % Tanggal (disetujui)
\newcommand\@bulan{Januari} % Bulan (disetujui)
\newcommand\@tahun{1970} % Tahun (disetujui)

% Sebuah langkah terakhir
\DeclareRobustCommand{\TanggalDisetujui}[3]{
    \renewcommand{\@tanggal}{#1}
    \renewcommand{\@bulan}{#2}
    \renewcommand{\@tahun}{#3}
}

% ++++++++++++++++++++++++++++
% KEPALA DEPARTEMEN
% Keterangan kepala departemen
% ++++++++++++++++++++++++++++

% Nama Kepala Departemen
\newcommand\@kadep{Nama Dosen} % Nama kepala departemen + gelar
\DeclareRobustCommand{\Kadep}[1]{
    \renewcommand{\@kadep}{#1}
}

% NIP Kepala Departemen
\newcommand\NIP@kadep{1122334455}
\DeclareRobustCommand{\NIPKadep}[1]{
    \renewcommand{\NIP@kadep}{#1}
}

% Nama Sekretaris Departemen
\newcommand\@sekdep{Nama Dosen}
\DeclareRobustCommand{\Sekdep}[1]{
    \renewcommand{\@sekdep}{#1}
}

% NIP Sekretaris Departemen
\newcommand\NIP@sekdep{1122334455}
\DeclareRobustCommand{\NIPSekdep}[1]{
    \renewcommand{\NIP@Sekdep}{#1}
}

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% DOSEN PEMBIMBING DAN PENGUJI
% Banyak dosen pembimbing menyesuaikan; saat ini, hanya satu (tahun 2024)
% Banyak dosen penguji menyesuaikan; saat ini, tiga (tahun 2024) 
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% Para pembimbing
\newcommand\pbb@satu{Nama Pembimbing Pertama} % Pertama
\newcommand\pbb@dua{} % Kedua
\newcommand\pbb@tiga{} % Ketiga
\DeclareRobustCommand{\DosenPembimbing}[3]{
    \renewcommand{\pbb@satu}{#1}
    \renewcommand{\pbb@dua}{#2}
    \renewcommand{\pbb@tiga}{#3}
}

% NIP Pembimbing
\newcommand\NIP@pbb@satu{1122334455}
\newcommand\NIP@pbb@dua{1122334455}
\newcommand\NIP@pbb@tiga{1122334455}
\DeclareRobustCommand{\NIPPembimbing}[3]{
    \renewcommand{\NIP@pbb@satu}{#1}
    \renewcommand{\NIP@pbb@dua}{#2}
    \renewcommand{\NIP@pbb@tiga}{#3}
}

% Para penguji
\newcommand\uji@satu{Nama Penguji Pertama}
\newcommand\uji@dua{Nama Penguji Kedua}
\newcommand\uji@tiga{Nama Penguji Ketiga}
\DeclareRobustCommand{\DosenPenguji}[3]{
    \renewcommand{\uji@satu}{#1}
    \renewcommand{\uji@dua}{#2}
    \renewcommand{\uji@tiga}{#3}
}

% NIP Penguji
\newcommand\NIP@uji@satu{1122334455}
\newcommand\NIP@uji@dua{1122334455}
\newcommand\NIP@uji@tiga{1122334455}
\DeclareRobustCommand{\NIPPenguji}[3]{
    \renewcommand{\NIP@uji@satu}{#1}
    \renewcommand{\NIP@uji@dua}{#2}
    \renewcommand{\NIP@uji@tiga}{#3}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FORMAT KHUSUS
% Desain format untuk pengetikan TA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Warna biru ITS
\definecolor{itsblue}{RGB}{0,103,172} % Dalam format RGB saja

% ++++++++++++++++++++++++++++++++++++++++
% FORMAT BAGIAN BUKU
% Mengatur desain susunan buku dan lainnya
% ++++++++++++++++++++++++++++++++++++++++

% Kebutuhan autoref
\def\indexautorefname{Indeks}
\def\figureautorefname{Gambar}
\def\tableautorefname{Tabel}
\def\appendixautorefname{Lampiran}
\def\chapterautorefname{Bab}
\def\sectionautorefname{Bagian}
\def\subsectionautorefname{Subbagian}
\def\equationautorefname{Persamaan}
\def\algorithmautorefname{Algoritma}

% Kustomisasi penamaan bagian buku, reupdate buat babel
\addto{\captionsenglish}{%
    \renewcommand{\contentsname}{Daftar Isi}
    \renewcommand\listfigurename{Daftar Gambar}
    \renewcommand\listtablename{Daftar Tabel}
    \renewcommand\listalgorithmname{Daftar Algoritma}
    \renewcommand\indexname{Indeks}
    \renewcommand\figurename{Gambar}
    \renewcommand\tablename{Tabel}
    \renewcommand\chaptername{Bab}
    \renewcommand\appendixname{Lampiran}
    \floatname{algorithm}{Algoritma}
}

% Penampilan judul bab dan lainnya
% Format Judul
\titleformat{\chapter}[block]
    {\centering\normalfont\bfseries\Large}{\MakeUppercase{\chaptername} \Roman{chapter}}{0pt}{\vskip 2pt \MakeUppercase{#1}}

\titlespacing*{name=\chapter}
    {0pt}{*0.001}{*10}
    
\titlespacing*{name=\chapter,numberless}
    {0pt}{*0.001}{*3}

% \patchcmd{\@makechapterhead}{\vspace*{50\p@}}{}{}{} % Hilangkan spasi sebelum \chapter
% \patchcmd{\@makeschapterhead}{\vspace*{50\p@}}{}{}{} % Hilangkan spasi sebelum \chapter*

\titlecontents{chapter}
    [0em] % Left margin
    {\vspace{3pt}\bfseries} % Above-code
    {\@chapapp\ \thecontentslabel\quad} % Numbered format
    {} % Numberless format
    {\titlerule*[1pc]{.}\contentspage} % Page number format
    [\vspace{3pt}] % Below-code

\g@addto@macro\appendices{%
    \addtocontents{toc}{\protect\renewcommand{\protect\@chapapp}{\appendixname}}%
}

% Format Section
\titleformat{\section}[block]
    {\normalfont\bfseries\large}{\thesection}{20pt}{#1}

\titleformat{name=\section,numberless}
    {\normalfont\bfseries\large}{}{0pt}{#1}

\titlespacing*{\section}
    {0pt}{*1}{*1}

\fancypagestyle{proposal}{
    % Settings for kertas biasa (proposal)
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyhead[L]{}
    \fancyhead[R]{}
    \fancyfoot[L]{}
    \fancyfoot[C]{\thepage}
    \fancyfoot[R]{}
}

% \fancypagestyle{proposal}{
%     \renewcommand{\headrulewidth}{0pt}
%     \renewcommand{\footrulewidth}{0pt}
%     \fancyhf{}
%     \fancyfoot[LE]{\thepage} % genap kiri
%     \fancyfoot[RO]{\thepage} % ganjil kanan
% }

\titleformat{name=\chapter,numberless}
    {\centering\normalfont\large\bfseries}{}{0pt}{\addcontentsline{toc}{chapter}{#1}\MakeUppercase{#1}}

% TikZ and PGF settings
\pgfplotsset{compat = newest}

% Kasih pembatas antar kata, mau tambahin tinggal define sendiri
\hyphenation{ber-sa-ma se-mes-ter ter-ki-ni da-lam di-bu-at pro-fe-sio-nal meng-em-bang-kan me-re-ka di-na-mi-ka ling-kung-an be-ker-ja sub-di-rek-to-rat ko-or-di-na-si per-ku-li-ah-an ber-sa-ma ker-ja te-pat me-ning-kat-kan di-pe-la-ja-ri ka-re-na me-ru-pa-kan peng-ga-bung-an pen-di-dik-an pro-gram bi-ro-kra-si bi-dang fung-si ke-ma-ha-sis-wa-an ad-mi-nis-tra-si ma-sih ter-ka-it a-khir me-nye-leng-ga-ra-kan ke-pa-da meng-gu-na-kan sen-di-ri-an pe-nu-lis ke-pu-tih su-ko-li-lo di-pu-bli-ka-si-kan di-se-but-kan di-be-nar-kan a-da-lah ber-man-fa-at da-ri pe-lak-sa-na-an cu-kup ma-ha-sis-wa me-nya-pa ber-tu-kar di-ba-gi-kan kar-ya e-fi-si-en-si ba-nyak da-hu-lu me-nu-rut ma-ta no-mor de-ngan se-hing-ga ser-ver glo-bal je-pang kon-sep ke-te-ram-pi-lan ku-ri-ku-lum se-per-ti}

% Some values
\newcommand{\dissthicc}{3pt}
\newcommand{\disstall}{35pt}
\newcommand{\disstalls}{20pt}

% Komponen pembantu
\newcommand{\katakunci}[1]{%
    \vspace*{1ex}
    \noindent\\
    \begin{tabularx}{\textwidth}{@{}lX@{}}
        \textbf{Kata kunci:} & \textit{#1}
    \end{tabularx}
}

\newcommand{\keywords}[1]{%
    \vspace*{1ex}
    \noindent\\
    \begin{tabularx}{\textwidth}{@{}lX@{}}
        \textbf{Keywords:} & \textit{#1}
    \end{tabularx}
}

% Kebutuhan penulisan algoritma
\newcommand{\Data}{\Statex \hspace{-\algorithmicindent} \textbf{Masukan: }}
\newcommand{\Result}{\Statex \hspace{-\algorithmicindent} \textbf{Luaran: }}
\newcommand\mycommfont[1]{\footnotesize\ttfamily\textcolor{DarkGreen}{#1}}

\renewcommand{\ALG@name}{Algoritma}
\newenvironment{breakablealgorithm}
    {% \begin{breakablealgorithm}
        \vspace{1mm}
        \refstepcounter{algorithm}% New algorithm
        \hrule height.8pt depth0pt \kern2pt% \@fs@pre for \@fs@ruled 
        \renewcommand{\caption}[2][\relax]{% Make a new \caption
        {\raggedright\textbf{\ALG@name~\thealgorithm} ##2\par}%
        \ifx\relax##1\relax % #1 is \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##2}%
        \else % #1 is not \relax
         \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##1}%
        \fi
        \kern2pt\hrule\kern2pt
        }
    }{% \end{breakablealgorithm}
        \kern2pt\hrule\relax% \@fs@post for \@fs@ruled 
        \vspace{2mm}
    }

\let\ps@plain\ps@fancy
  
% Miscellaneous
\newcommand{\var}{\texttt}
\definecolor{itsblue}{RGB}{0, 103, 172}
\newcommand\fillin[1][3cm]{\makebox[#1]{\dotfill}}

% Ukuran gaya penulisan & spasi
\def\regularfontsize{12} % Ukuran tulisan biasa
\def\coverfontsize{14} % Ukuran tulisan sampul
\def\titlefontsize{18} % Ukuran tulisan judul
\def\fontbaselineskip{2} % Spasi antar baris

% Berkas-berkas penting
\newcommand{\FrontMatter}{\input{front-matter/000_front-matter}}

% Pengaturan tata letak kata pada buku (biar gak tabrak sana sini)
\tolerance=1
\emergencystretch = \maxdimen
\hyphenpenalty=100000
\hbadness=1000

% URL settings
\g@addto@macro{\UrlBreaks}{\UrlOrds}