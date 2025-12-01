# ML Project Strategy Document

## Project Title
*Fine-tuning Gemma 2 for Spanish QA using SQUAD-es and MLQA*

---

## 1. Motivation & Objectives

- **Background:**  
  Most open-source LLM variants are primarily trained in English. This project demonstrates how to adapt a strong base LLM —Gemma 2— for improved performance in Spanish QA.  
  Although inspired by the Kaggle Gemma Language Tuning competition, the true aim is not to participate in the competition itself, but to document and showcase the complete LLM adaptation workflow for the purpose of learning and showcasing skills.

- **Project Goals:**  
  - Fine-tune Gemma 2 for Spanish language QA.
  - Build a reproducible, modular ML solution: data curation, training, evaluation, and serving.
  - Apply and document ML engineering practices (version control, experiment tracking, testing, reproducibility, deployment).

- **Model Variant Choice:**  
  Gemma is available in multiple sizes (2B, 7B, 9B, 27B). For this project:
  - Gemma 2B will be used for initial development, prototyping, and experimentation, given its lower resource requirements and sufficient capacity for demonstrating the fine-tuning workflow.
  - If pipeline and resource constraints permit, results will be extended to Gemma 7B to benchmark improvements and analyze scalability.

- **Intended Audience:**  
  - Technical recruiters
  - ML/AI engineers and practitioners
  - Anyone interested in practical LLM adaptation

---

## 2. Problem Statement & Success Criteria

- **Task Definition:**  
  Fine-tune the Gemma 2 LLM to improve its general Spanish language understanding and generation.  
  Focus: Adapt the existing English-centric Gemma 2 model to perform better on Spanish QA.  
  The project does not focus on a specific dialect, but aims for broad Spanish language capability.

- **Success Metrics:**  
  - **Quantitative:**  
    - Primary: F1 and Exact Match (EM)  
    - Auxiliary: VRAM usage, inference latency, throughput  
    - Optional: Benchmark on gold standard sets  
    - (If SQuAD v2 added) Precision/recall for “no answer” detection, AUROC for answerability classification
  - **Qualitative:**  
    - Manual review of generated outputs (fluency, correctness, relevance)  
    - Basic error analysis: common mistakes or failure modes (incorrect grammar, Anglicisms, misunderstood context)  
    - Optional: Human review (native speaker) of sampled outputs

- **Scope:**  
  - **In-Scope:**  
    - Spanish data curation/cleaning
    - Fine-tuning via supervised learning (SFT/QLoRA based on resource constraints)  
    - Evaluation on general Spanish tasks (QA)
    - Model serving and basic performance benchmarks
    - 2B model for all steps; 7B stretch goal/ablation if resources allow 
  - **Out-of-Scope:**  
    - Domain-specific adaptation (medical/legal/technical or specific dialect Spanish)  
    - RLHF-style reinforcement learning  
    - Larger Gemma models (9B, 27B—not practical for current hardware)
    - Large-scale human evaluation or robust bias/fairness analysis

---

## 3. Datasets

- **Data Sources:**  
  - **Training:**  
    - SQuAD-es v1.1 (machine-translated Spanish version of SQuAD 1.1 by Casimiro Pio).  
      [Hugging Face: ccasimiro/squad_es](https://huggingface.co/datasets/ccasimiro/squad_es)
  - **Evaluation/Benchmarking:**  
    - Spanish subset of MLQA ([facebook/mlqa](https://huggingface.co/datasets/facebook/mlqa))
    - Spanish subset of XQuAD ([google/xquad](https://huggingface.co/datasets/google/xquad))
  - **Bonus/Robustness Evaluation:**  
    - SQuAD-es v2 (for potential “no answer” capability, same translation source as v1.1)
    - TyDi QA (Spanish, gold passage benchmark, [google-research-datasets/tydiqa](https://huggingface.co/datasets/google-research-datasets/tydiqa))
    - PAWS-X (Spanish, paraphrase pairs for ablation/robustness, [google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x))
  - **Licensing/Ethics:**  
    - All datasets are publicly available for research use. I will review their respective licenses and ensure compliance.  
    - No private or sensitive user data is included in any dataset.

---

## 4. Timeline & Milestones

| Milestone             | Description                                  | Timeline  |
|---------------------- |----------------------------------------------|-----------|
| Strategy Doc Complete | Planning, scoping, and design ready          | 1-2 days  |
| Data Ready            | Curation, cleaning, preprocessing complete   | 1 week    |
| Baseline Model        | First training run, sanity checks            | 1 week    |
| Fine-tuned Model      | Main experiment(s) complete                  | 1-2 weeks |
| Evaluation            | Metrics, error analysis, result reporting    | 1 week    |
| Serving/Demo          | Inference endpoint or demo runs              | 1 week    |
| Final Polish          | Docs, code cleanup, repo structuring         | 1-2 days  |

---

## 5. Deliverables

- Documented notebook (end-to-end: data, training, eval, serving)
- Model artifacts (checkpoint, config, logs)
- Demo (notebook cell or web demo)
- README with concise overview and instructions

---

## 6. References/Bibliography

[Google Gemma Global Communication Comp Announcement](https://discuss.ai.google.dev/t/join-the-competition-unlock-global-communication-with-gemma/47814)
[Kaggle Gemma Language Tuning Comp](https://www.kaggle.com/competitions/gemma-language-tuning)

---
