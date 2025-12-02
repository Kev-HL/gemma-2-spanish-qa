
# Fine-tuning Gemma 2 for Spanish QA using SQUAD-es and MLQA

Most open-source LLM variants are primarily trained in English. This project demonstrates how to adapt a strong base LLM —Gemma 2— for improved performance in Spanish QA.  
Although inspired by the Kaggle Gemma Language Tuning competition, the true aim is not to participate in the competition itself, but to document and showcase the complete LLM adaptation workflow for the purpose of learning and showcasing skills.

---

## Project Overview

(...)

---

## Key Results

(...)

---

## Dataset

- **Training:**  
- SQuAD-es v1.1 (machine-translated Spanish version of SQuAD 1.1 by Casimiro Pio).  
  [Hugging Face: ccasimiro/squad_es](https://huggingface.co/datasets/ccasimiro/squad_es)
  The SQuAD-es dataset is licensed under the *CC BY 4.0* license, the code in the translation repository is licensed under the *GNU GPLv3*.
- **Evaluation/Benchmarking:**  
- Spanish subset of MLQA ([facebook/mlqa](https://huggingface.co/datasets/facebook/mlqa))
  The dataset, which is derived from paragraphs in Wikipedia, is licensed under *CC-BY-SA 3.0*. The code in the repository is licensed according to *Attribution-NonCommercial 4.0 International*.
- Spanish subset of XQuAD ([google/xquad](https://huggingface.co/datasets/google/xquad))
  This dataset is distributed under the *CC BY-SA 4.0* license.
- **Bonus Evaluation:**  
- SQuAD-es v2 (for potential “no answer” capability, same translation source as v1.1, same licensing)
- PAWS-X (Spanish, paraphrase pairs for ablation/robustness, [google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x))
  Free use license, data sourced from Google LLC.

_Note: Copies of the LICENSE files are provided in their respective datasets folder on **data/**_  

---

## Tech Stack

- Python
- PyTorch
(...)

---

## Folder Structure

```
data/       # Dataset and annotations
(...)
```

---

## How to Run

(...)

---

## Citations & References

**Datasets:**
- SQuAD (original): [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) (Rajpurkar, Zhang, Lopyrev, Liang. EMNLP 2016)
- SQuAD (translation): [Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering](https://arxiv.org/abs/1912.05200) (Carrino, Costa-jussà, Fonollosa. arXiv 2019)
- MLQA: [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://arxiv.org/abs/1910.07475) (Lewis, Oguz, Rinott, Riedel, Schwenk. arXiv 2019)
- XQuAD: [On the cross-lingual transferability of monolingual representations](https://arxiv.org/abs/1910.11856) (Artetxe, Ruder, Yogatama. arXiv 2019)
- PAWS: [PAWS: Paraphrase Adversaries from Word Scrambling](https://arxiv.org/abs/1904.01130) (Zhang, Baldridge, He. NAACL 2019)
- PAWS-X: [PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification](https://arxiv.org/abs/1908.11828) (Yang, Zhang, Tar, Baldridge. EMNLP 2019)

**Kaggle competition:**
[Google Gemma Global Communication Comp Announcement](https://discuss.ai.google.dev/t/join-the-competition-unlock-global-communication-with-gemma/47814)
[Kaggle Gemma Language Tuning Comp](https://www.kaggle.com/competitions/gemma-language-tuning)

---

## Contact

For questions reach out via GitHub (Kev-HL).
