# MIN-K% Prob — Detecting Pretraining Data from LLMs

<p align="center">
  <img src="https://img.shields.io/badge/Course-Reinforcement%20Learning-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Institution-NED%20University-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Paper-ICLR%202024-purple?style=flat-square"/>
</p>

> Reproduction and critical analysis of the ICLR 2024 paper:  
> **"Detecting Pretraining Data from Large Language Models"** — Shi et al., 2024  
> [[Paper PDF (local)]](assets/2310.16789v3.pdf) · [[arXiv]](https://arxiv.org/abs/2310.16789) · [[Official Repo]](https://github.com/swj0419/detect-pretrain-code) · [[WikiMIA Dataset]](https://huggingface.co/datasets/swj0419/WikiMIA)

---

## What Is This?

This project is a course assignment that **reproduces, analyses, and extends** the **MIN-K% Prob** membership inference method. Given a piece of text and black-box access to an LLM, MIN-K% Prob determines whether that text was part of the model's pretraining data — without needing any reference model or access to the training corpus.

### Core Idea

> An unseen text tends to contain a few outlier tokens with unexpectedly **low probability** under the LLM.  
> A seen (memorised) text is **less likely** to have such low-probability outliers.

MIN-K% Prob selects the bottom `k%` tokens by probability and averages their log-likelihood as the detection score.

```
log p closer to 0   →  model confident on hard tokens  →  MEMBER  (seen)
log p very negative →  model struggles on hard tokens  →  NON-MEMBER (unseen)
```

---

## Repository Structure

```
ResearchAI/
├── assets/
│   └── 2310.16789v3.pdf                         # Original ICLR 2024 paper
├── Detecting Pretraining Data From LLMs.ipynb   # Main notebook (all experiments)
├── requirements.txt                             # Python dependencies
├── CITATION.cff                                 # How to cite this repo
├── LICENSE                                      # MIT License
├── .gitignore                                   # Files excluded from Git
└── README.md                                    # This file
```

---

## Notebook Walkthrough

The notebook is fully self-contained and structured into five progressive experiments:

| # | Section | Models Used | Key Output |
|---|---------|-------------|------------|
| 1 | **Toy Memorisation Experiment** | `pythia-70m` | MIN-K% successfully detects a fine-tuned essay vs. unseen control text |
| 2 | **MIA Method Comparison** | `pythia-70m` | MIN-K% outperforms PPL and Zlib baselines on hand-crafted dataset |
| 3 | **WikiMIA Benchmark Evaluation** | `pythia-2.8b` | AUC = 0.5956 vs. paper's 0.67 (gap explained by model/compute constraints) |
| 4 | **Hyperparameter Sweep (k%)** | `pythia-2.8b` | Optimal k = 10 for our setup (paper reports k = 20 for larger models) |
| 5 | **Multi-Model Scaling Study** | `pythia-70m`, `160m`, `410m` | AUC and detection quality improve with model size and text length |

---

## Results

### WikiMIA Evaluation (Section 3)

| Method | Our AUC | Paper AUC (Pythia-2.8B) |
|--------|:-------:|:-----------------------:|
| Perplexity (PPL) | — | 0.61 |
| Zlib Entropy | — | 0.65 |
| **MIN-K% Prob** | **0.5956** | **0.67** |

> Our gap from the paper's result is expected — we ran on free-tier Google Colab (T4 GPU) with 100 samples. The paper used full WikiMIA with larger infrastructure.

### Key Findings

- MIN-K% consistently **outperforms PPL and Zlib** baselines — matches the paper's claims
- Detection quality **increases with model size** — larger models memorise more, making detection easier
- Detection quality **increases with text length** — more tokens provide a stronger signal
- The optimal `k` is **sensitive to model scale** — we found `k=10` best for smaller models; paper found `k=20` for 2.8B+

---

## Setup & Installation

### Option A — Google Colab (Recommended, Zero Setup)

The notebook installs all dependencies automatically in the first cell:

```python
!pip install -q transformers datasets accelerate
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Tip:** For the WikiMIA sections, switch to a T4 GPU runtime in Colab (`Runtime → Change runtime type → T4 GPU`) for faster inference.

### Option B — Local Setup

**Requirements:** Python 3.10+, and a CUDA GPU for the larger Pythia models.

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ResearchAI.git
cd ResearchAI

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook "Detecting Pretraining Data From LLMs.ipynb"
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.1.0 | Model inference and log-probability computation |
| `transformers` | ≥ 4.40.0 | Loading Pythia models and tokenisers |
| `datasets` | ≥ 2.18.0 | Loading WikiMIA benchmark from Hugging Face Hub |
| `accelerate` | ≥ 0.27.0 | Multi-GPU / device-map support |
| `scikit-learn` | ≥ 1.4.0 | ROC curve and AUC computation |
| `numpy` | ≥ 1.26.0 | Numerical operations |
| `matplotlib` | ≥ 3.8.0 | Plotting ROC curves and AUC bar charts |
| `zlib` | stdlib | Zlib entropy baseline scoring |

---

## Models

All models are loaded automatically from [Hugging Face Hub](https://huggingface.co/EleutherAI). No manual download needed.

| Model | Params | Used In |
|-------|-------:|---------|
| [`EleutherAI/pythia-70m`](https://huggingface.co/EleutherAI/pythia-70m) | 70M | Toy experiment, MIA comparison, scaling sweep |
| [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m) | 160M | Scaling sweep |
| [`EleutherAI/pythia-410m`](https://huggingface.co/EleutherAI/pythia-410m) | 410M | Scaling sweep |
| [`EleutherAI/pythia-2.8b`](https://huggingface.co/EleutherAI/pythia-2.8b) | 2.8B | WikiMIA evaluation, hyperparameter sweep |

---

## Dataset

**[WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA)** — the dynamic benchmark introduced in the paper.

| Split | Description |
|-------|-------------|
| Member data | Wikipedia articles created **before 2017** — seen during Pythia pretraining |
| Non-member data | Wikipedia event articles created **after Jan 2023** — guaranteed unseen |
| Lengths used | `WikiMIA_length32`, `WikiMIA_length64`, `WikiMIA_length128` |

```python
from datasets import load_dataset
dataset = load_dataset("swj0419/WikiMIA", split="WikiMIA_length64")
```

---

## The Algorithm

```
Input: text x = [x₁, x₂, ..., xₙ]  +  target LLM  +  k (default = 20%)
─────────────────────────────────────────────────────────────────────────
Step 1  Feed x through the LLM (teacher forcing)
        → log p(xᵢ | x₁,...,xᵢ₋₁)  for every token i

Step 2  Sort tokens by log probability (ascending)
        → Identify the bottom k% — the "hardest" tokens

Step 3  Average the log-likelihoods of those k% tokens
        → MIN-K% PROB(x) = (1/|Min-K%(x)|) × Σ log p(xᵢ | context)

Decision  score > threshold  →  MEMBER (seen during training)
          score < threshold  →  NON-MEMBER (not seen)
─────────────────────────────────────────────────────────────────────────
```

Full mathematical formulation is in Section 3 of the paper: [`assets/2310.16789v3.pdf`](assets/2310.16789v3.pdf)

---

## How to Cite

If you use this code or notebook, please cite:

```bibtex
@software{hassan_ahmed_minkprob_2026,
  author    = {Hassan, Messum and Ahmed, Uzair},
  title     = {MIN-K\% Prob: Detecting Pretraining Data from LLMs — Reproduction \& Analysis},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/<your-username>/ResearchAI}
}
```

And the original paper:

```bibtex
@inproceedings{shi2024detecting,
  title     = {Detecting Pretraining Data from Large Language Models},
  author    = {Shi, Weijia and Ajith, Anirudh and Xia, Mengzhou and Huang, Yangsibo
               and Liu, Daogao and Blevins, Terra and Chen, Danqi and Zettlemoyer, Luke},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2310.16789}
}
```

---

## Authors

| Name | Roll No. | Contribution |
|------|----------|-------------|
| **Messum Hassan** | AI-043 | Implementation, experiments, code |
| **Uzair Ahmed** | AI-012 | Report writing, documentation, analysis |

**Course:** Reinforcement Learning  
**Department:** Computer Science and Information Technology  
**Institution:** NED University of Engineering and Technology  
**Batch:** 2022–2026

---

## License

This project is licensed under the [MIT License](LICENSE).  
The MIN-K% method and WikiMIA benchmark are credited to the original authors (Shi et al., ICLR 2024).  
The paper PDF in `assets/` is included for academic reference only under fair use.
