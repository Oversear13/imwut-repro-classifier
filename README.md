# Reproducibility Classification of IMWUT Papers with MLPs and a Local LLM

This repository accompanies the bachelor thesis

> **A Deep Learning and Local LLM approach to Reproducibility Classification of Scientific Publications using a Self-collected Dataset**
> Mike Specht, University of Siegen, Department of Electrical Engineering and Computer Science, October 2026.
> First examiner: Univ.-Prof. Dr. Kristof van Laerhoven. Second examiner: Robin Burchard.

It contains a manually annotated dataset on the availability of reproducibility-related artifacts in research papers, and the code of two local pipelines that predict these properties from the full text of a paper.

## Contents

| Path | Description |
|---|---|
| `IMWUT_Dataset_Original.csv` | **Ground Truth Dataset**: all manual annotations with the original label values |
| `IMWUT_Dataset_klassifikation.csv` | **Classification Dataset**: binarized targets derived from the Ground Truth Dataset, as used by the pipelines |
| `01_cockpit.py` | Shared configuration: parameters, paths, prediction targets, binarization rules, text extraction, chunking, and Ollama helpers |
| `Ollama_Embeddings.ipynb` | Pipelines A and B: embedding, pooling, MLP training and evaluation, LLM classification and evaluation |
| `requirements.txt` | Python libraries used by the code, with exact versions |

## The datasets

### Ground Truth Dataset

The Ground Truth Dataset covers all **655 research papers** published in the *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies* (IMWUT) from **2023 to 2025**. Each paper was annotated by one of three annotators (Felix Schneider, Mike Specht, Haitham El Euch) according to a shared annotation schema. Per paper, it records:

- **Metadata**: citation, title, year, issue, article number, DOI URL, number of pages, and the `paper_id` (format `YYYY_Mon_NNN`, e.g. `2025_Mar_010`), which serves as the primary key.
- **Paper type**: one or more of *Algorithm Development*, *Hardware Development*, *Software Development*, *User Study*, *Dataset Paper*, and *Literature Review*.
- **Reproducibility labels**: whether code, data, specific hardware, or proprietary software are required; whether code and data are available; the links to code and data; and further notes.

The full annotation schema and its definitions are described in Section 3 of the thesis.

### Classification Dataset

The Classification Dataset contains **654 papers** and **14 binary targets** (`Yes`/`No`):

| Column | Description |
|---|---|
| `paper_id` | Primary key, links to the Ground Truth Dataset |
| `year` | Year of publication |
| `paper_type` | Paper types, separated by semicolons (encoded as six binary targets) |
| `code_required`, `code_available`, `code_url` | Code is required / available / linked |
| `data_required`, `data_available`, `data_url` | Data is required / available / linked |
| `hardware_required` | Specific hardware beyond standard computing is required |
| `software_required` | Proprietary, paid, or otherwise non-open-source software is required |

The binarization rules are given in the thesis (Table *Binarization of the label fields*).

### Papers are not included

The PDFs of the papers are **not part of this repository** for copyright reasons. They can be downloaded from the [ACM Digital Library](https://dl.acm.org/journal/imwut).

## The pipelines

Both pipelines run locally and receive the same text chunks (recursive character splitting, 1,200 characters with 200 characters overlap, reference sections removed). The text is taken either from the raw PDF text (PyMuPDF) or from a Markdown conversion (pymupdf4llm).

- **Pipeline A** embeds each chunk with `nomic-embed-text-v2-moe`, pools the chunk embeddings into one vector per paper (mean, max, or mean–max concatenation), and classifies it with multi-layer perceptrons (scikit-learn and PyTorch) in a stratified multi-label cross-validation.
- **Pipeline B** classifies each chunk with the LLM **Llama 3.2** (3.2B, Q4_K_M) using a few-shot prompt and aggregates the chunk answers into one prediction per paper. The full prompt is given in the appendix of the thesis.


## Running the code

### Requirements

- Python 3.14
- [Ollama](https://ollama.com/) 0.24.0, running locally at `http://localhost:11434`, with the models
  ```bash
  ollama pull nomic-embed-text-v2-moe
  ollama pull llama3.2
  ```
- The Python libraries in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

The experiments in the thesis were run on an Intel Core i7-7800X with 24 GB RAM and an AMD Radeon RX 6800 XT under Bazzite 44, with Ollama in a Podman container with GPU support. The MLPs were trained on the CPU.

### Folder layout

**`01_cockpit.py` must be in the same directory as the notebook.** The notebook loads the configuration with `%run -i 01_cockpit.py`, which only finds the file in the current directory. In addition, `01_cockpit.py` defines all paths relative to the working directory (`BASE_DIR = Path(os.getcwd())`), so the notebook must be started from this directory as well. The `daten/` folder lies next to both files:

```
.
├── 01_cockpit.py
├── Ollama_Embeddings.ipynb
└── daten/
    ├── pdfs/            # PDFs of the papers, named by paper_id
    ├── raw_downloads/   # optional: unsorted downloads from the ACM DL
    ├── labels/
    │   └── IMWUT_Dataset_klassifikation.csv
    ├── markdown/        # created: Markdown conversions
    ├── embeddings/      # created: chunk embeddings
    ├── predictions_llm/ # created: LLM answers
    └── ...              # further output folders are created automatically
```

### Steps

1. Copy `IMWUT_Dataset_klassifikation.csv` into `daten/labels/`.
2. Download the PDFs and place them in `daten/raw_downloads/`. `01_cockpit.py` provides a function that sorts them into `daten/pdfs/` and renames them by `paper_id` (based on the DOI in the ACM reference format).
3. Choose the text source and pooling strategy at the top of `01_cockpit.py` (`DATEN_QUELLE = "markdown"` or `"pdf_raw"`, `POOLING_STRATEGIE`).
4. Start Jupyter in the directory that contains `01_cockpit.py`, then run `Ollama_Embeddings.ipynb` from top to bottom. It loads the configuration with `%run -i 01_cockpit.py`, then runs text extraction, chunking, embedding, the MLP experiments, and the LLM classification. Each run writes a report and its configuration to `daten/`.

All random seeds are fixed, and the LLM runs with temperature 0 and a fixed seed.
