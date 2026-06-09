
# Fine-tuning Gemma 2 for Spanish QA using SQUAD-es, MLQA and XQuAD

![CI](https://github.com/Kev-HL/gemma-2-spanish-qa/actions/workflows/ci.yaml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)
![PEFT](https://img.shields.io/badge/fine--tuning-PEFT-purple.svg)
![Weights & Biases](https://img.shields.io/badge/experiment%20tracking-W%26B-FFBE00.svg)


While Gemma 2 is primarily trained on English, it exhibits multilingual capabilities that benefit from targeted fine-tuning in specific languages. This project demonstrates how to adapt Gemma 2 for improved Spanish question-answering performance using parameter-efficient fine-tuning techniques.

The goal is to perform a complete fine-tuning workflow: from data preparation and model adaptation, through training and evaluation, to benchmarking against established baselines, showcasing both technical execution and learning in a focused, professional manner.

---

## Project Overview

This project fine-tunes **Gemma 2 2B**, a compact decoder-only LLM, for Spanish question-answering using **LoRA** (Low-Rank Adaptation)—enabling training within local hardware constraints (16GB VRAM).

**Research Question:** Can a small decoder-only model learn to perform Spanish extractive QA reasonably well with basic fine-tuning, despite architectural constraints?

**Scope:**
- Single model: Gemma 2 2B
- Single training dataset: Machine-translated SQuAD 1.1 (Spanish)
- Test sets: Spanish subsets of MLQA and XQuAD (held-out evaluation)
- Baseline comparison: mBERT
- Execution: Local only (no cloud compute or distributed training)

---

## Methodology

**Model Selection:** Gemma 2 2B was chosen as a balance between model capacity and hardware feasibility. While its multilingual capabilities are limited compared to Gemma 3, it remains a strong open base model suitable for adaptation. LoRA fine-tuning enables parameter-efficient training (Millions of trainable parameters instead of Billions) within the 16GB VRAM constraint.

**Training Approach:** The model is fine-tuned using a causal language modeling objective with LoRA adapters on the SQuAD-es training set. Hyperparameters were tuned to optimize validation F1 scores.

**Evaluation Metrics & Methodology:** F1 and Exact Match (EM) scores are computed using the standard SQuAD evaluation script (via HuggingFace's *evaluate* library). Evaluation is performed on SQuAD-es validation set during training and on held-out Spanish test sets (MLQA, XQuAD) for final benchmarking. Prior to training, data cleaning removes invalid entries and any overlap with test sets.

**Baseline Comparison:** mBERT serves as a reference baseline, a proven encoder-only model well-suited for extractive QA with multilingual support. A solid baseline fine-tuning (no hyperparameter iteration) on the same training data establishes performance expectations.

**Success Criteria:** The fine-tuning is considered successful if the adapted Gemma 2 model achieves F1 scores within a reasonable range of mBERT performance on held-out test sets, validating that decoder-only models can adapt to Spanish extractive QA despite architectural limitations. EM scores are expected to be substantially lower due to the generative (token-by-token) approach vs. extractive span selection.

---

## Key Results

**Gemma 2 Zero-shot Improvement**

| Dataset | Zero-shot F1 | Fine-tuned F1 | Improvement |
|---------|----------|-----------|-----|
| SQuAD Val | 31.61 | 56.53 | +24.9 |
| MLQA | 39.71 | 61.23 | +21.5 |
| XQuAD | 36.60 | 64.01 | +27.4 |

**Final Model Performance Summary**

| Dataset | mBERT F1 | Gemma 2 F1 | Gap |
|---------|----------|-----------|-----|
| SQuAD Val | 72.60 | 56.53 | −16.1 |
| MLQA | 64.31 | 61.23 | −3.1 |
| XQuAD | 73.53 | 64.01 | −9.5 |

All results, training curves, analysis and conclusions can be found on the **01_experiments.ipynb** notebook, located on the `notebooks/` folder.

---

## Key Findings

- Gemma 2 2B can be effectively fine-tuned for Spanish QA with modest computational resources, achieving 56.5 F1 on SQuAD (25-point improvement from zero-shot).

- Gemma 2 benefitted more to exposure to new data (format learning) than hyperparameter optimization.

- Despite architectural differences from mBERT, Gemma 2 reaches competitive performance on high-quality translated data (MLQA: −3.1 F1, XQuAD: −9.5 F1).

- The model excels at semantic understanding but struggles with exact matching (EM ~15 vs. mBERT's 56), a fundamental constraint of the generative approach.

- Answer grounding via fuzzy matching provides modest improvements and serves as practical post-processing for inference.

---

## Datasets

- **Training:**  
- SQuAD-es v1.1 (machine-translated Spanish version of SQuAD 1.1 by Casimiro Pio).  
  [HF: ccasimiro/squad_es](https://huggingface.co/datasets/ccasimiro/squad_es)  
  The SQuAD-es dataset is licensed under the *CC BY 4.0* license, the code in the translation repository is licensed under the *GNU GPLv3*.
- **Evaluation/Benchmarking:**  
- Spanish subset of MLQA ([HF: facebook/mlqa](https://huggingface.co/datasets/facebook/mlqa))  
  The dataset, which is derived from paragraphs in Wikipedia, is licensed under *CC-BY-SA 3.0*. The code in the repository is licensed according to *Attribution-NonCommercial 4.0 International*.
- Spanish subset of XQuAD ([HF: google/xquad](https://huggingface.co/datasets/google/xquad))  
  This dataset is distributed under the *CC BY-SA 4.0* license.

_Note: Copies of the LICENSE files are provided in their respective datasets folder on **data/**_  

---

## Tech Stack

**Core Framework & Training:**
- PyTorch — Deep learning framework
- Hugging Face Transformers — LLM implementation and pre-trained models
- PEFT — Parameter-efficient fine-tuning (LoRA)
- TRL — Trainer utilities for language models (SFTTrainer)

**Data & Evaluation:**
- Hugging Face Datasets — Data loading and preprocessing
- Hugging Face Evaluate — Standard evaluation metrics (F1, EM via SQuAD script)

**Experiment Tracking & Monitoring:**
- Weights & Biases — Experiment tracking, hyperparameter logging, and result comparison

**Utilities:**
- NumPy, Pandas — Data manipulation and analysis
- Matplotlib — Visualization
- Sentence-Transformers — Semantic similarity filtering during EDA
- Rapidfuzz — Fuzzy string matching for grounding generative answers to context spans

**Development:**
- Black — Code formatting
- Flake8 — Linting
- Pytest — Unit testing

---

## Folder Structure

```
.github/workflows/    # CI/CD pipeline configuration
artifacts/            # Model checkpoints and results (not tracked)
configs/              # Experiment and other configurations (JSON)
data/                 # Dataset and annotations (only license files tracked)
notebooks/            # Jupyter notebooks for exploration
scripts/              # Python executable scripts
src/                  # Python modules (functions, classes...)
tests/                # Test units
```

---

## How to Run

```bash
# 1. Clone repository
git clone https://github.com/Kev-HL/gemma-2-spanish-qa.git
cd gemma-2-spanish-qa

# 2. Set up credentials
#    Create .env file in project root with the following keys:
#
#    HF_TOKEN (Required)
#      - Gemma 2 models are gated; token required for access
#      - Get token: https://huggingface.co/settings/tokens
#      - Request access: https://huggingface.co/google/gemma-2-2b
#      - Add to .env: HF_TOKEN=hf_...
#
#    WANDB_API_KEY (Optional)
#      - Required only if using Weights & Biases logging
#      - Can be disabled in config: set "report_to": "none"
#      - Get API key: https://wandb.ai/settings#apikeys
#      - Add to .env: WANDB_API_KEY=wandb_v1_...

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. [OPTIONAL] Download raw datasets
#    (EDA notebook auto-downloads if not present)
make download_data

# 5. Run EDA notebook to prepare datasets
jupyter notebook notebooks/
# Open 00_EDA.ipynb and run all cells to clean and prepare datasets

# 6. Run training or evaluation
python <script_path> <config_path>

# Examples:
python scripts/training_gemma2.py configs/training/gemma2_final_model.json
python scripts/training_mbert.py configs/training/mbert_baseline.json
python scripts/evaluation_gemma2.py configs/eval/gemma2_final_squad.json
python scripts/evaluation_mbert.py configs/eval/mbert_baseline_mlqa.json

# Valid configuration schemas: configs/schemas/
```

---

## Citations & References

**Models and methods:**
- Gemma 2 2B: [Hugging Face Model Page](https://huggingface.co/google/gemma-2-2b)
- mBERT: [Hugging Face Model Page](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al. ICLR 2022)
- rsLoRA: [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732) (Kalajdzievski. arXiv 2023)

**Datasets:**
- SQuAD (original): [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) (Rajpurkar, Zhang, Lopyrev, Liang. EMNLP 2016)
- SQuAD (translation): [Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering](https://arxiv.org/abs/1912.05200) (Carrino, Costa-jussà, Fonollosa. arXiv 2019)
- MLQA: [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://arxiv.org/abs/1910.07475) (Lewis, Oguz, Rinott, Riedel, Schwenk. arXiv 2019)
- XQuAD: [On the cross-lingual transferability of monolingual representations](https://arxiv.org/abs/1910.11856) (Artetxe, Ruder, Yogatama. arXiv 2019)

**Kaggle competition (inspiration):**
- [Google Gemma Global Communication Comp Announcement](https://discuss.ai.google.dev/t/join-the-competition-unlock-global-communication-with-gemma/47814)
- [Kaggle Gemma Language Tuning Comp](https://www.kaggle.com/competitions/gemma-language-tuning)

**Sentence Transformers (sentence similarity):**  
- [SentenceTransformer](https://www.sbert.net/)
- [SentenceTransformer Pretrained Models Leaderboard](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (Reimers, Gurevich, EMNLP 2019)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) (Muennighoff, Tazi1, Magne1, Reimers. arXiv 2022)

**Main Tools & Libraries:**
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Weights & Biases](https://wandb.ai/)

---

## Contact

For questions reach out via GitHub (Kev-HL).
