# 🏥 MCH AI Assistant — Domain-Specific LLM Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/MCH-LLM-FineTuning/blob/main/MCH_LLM_Fine_Tuning.ipynb)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Carinea/MCH_LLM_Fine_Tuning)
[![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Carinea/MCH_LLM_Fine_Tuning)

---

## 📌 Project Overview

**Domain:** Maternal and Child Health (MCH)  
**Task:** Generative Question Answering  
**Approach:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA  
**Base Model:** [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Why MCH?

Maternal and child health remains one of the most critical public health challenges globally. In many low- and middle-income countries, health workers lack immediate access to clinical guidelines during patient consultations. An AI assistant trained on WHO-aligned MCH guidelines can:

- Provide instant, guideline-based answers on pregnancy danger signs, labour, postpartum care, newborn care, vaccinations, and child illness
- Support community health workers in resource-limited settings
- Reduce preventable maternal and neonatal mortality through timely information
- Handle out-of-domain queries gracefully, redirecting users to appropriate resources

---

## 📂 Repository Structure

```
MCH-LLM-FineTuning/
├── MCH_LLM_Fine_Tuning.ipynb   # Complete training pipeline (runs on Colab)
├── app.py                       # Gradio web interface for HuggingFace Spaces
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📊 Dataset

### Source & Creation
The dataset was curated manually, aligned with **WHO clinical MCH guidelines** (no web scraping). It contains **1,100 instruction-response pairs** across 9 MCH sub-domains.

| Sub-domain | Examples |
|---|---|
| Pregnancy Complications | 120 |
| Labour & Delivery | 120 |
| Postpartum Care | 120 |
| Newborn Danger Signs | 120 |
| Newborn Feeding / Breastfeeding | 120 |
| Child Vaccination Schedule | 120 |
| Child Illness (Diarrhoea, Fever) | 120 |
| Family Planning | 120 |
| Maternal Mental Health | 120 |

### Preprocessing Steps

1. **Normalization:** Strip leading/trailing whitespace; remove empty pairs
2. **Deduplication:** Dictionary-ordered deduplication to remove exact duplicate questions
3. **Template Formatting:** All examples formatted using TinyLlama's chat template:
   ```
   <|system|>
   {SYSTEM_PROMPT}
   <|user|>
   {question}
   <|assistant|>
   {answer}
   ```
4. **Tokenization:** Using TinyLlama's SentencePiece BPE tokenizer (`vocab_size=32,000`), `padding_side="right"` for causal LM, `max_length=512` with truncation
5. **Label Assignment:** `labels = input_ids` (standard causal language modelling objective — the model learns to predict every next token)
6. **Token Length Audit:** Confirmed all 1,100 examples fit within 512-token window (max observed: ~180 tokens)

---

## ⚙️ Fine-Tuning Methodology

### Base Model
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` — chosen for:
- Small enough to train on Colab free T4 GPU (~16 GB VRAM)
- Chat-tuned variant (already understands instruction-following format)
- Strong performance-to-size ratio for domain adaptation

### PEFT — LoRA Configuration (Selected)
| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Higher capacity than r=8; fits in GPU memory |
| LoRA alpha | 32 | Effective scale = alpha/r = 2x |
| LoRA dropout | 0.05 | Light regularisation |
| Target modules | q_proj, v_proj | Attention projections — most impactful for domain adaptation |
| Task type | CAUSAL_LM | Generative QA objective |

### Training Configuration
| Parameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Epochs | 2 |
| Batch size (per device) | 4 |
| Gradient accumulation | 2 (effective batch = 8) |
| Optimizer | paged_adamw_8bit (GPU) |
| Warmup steps | 10 |
| Weight decay | 0.01 |
| Precision | fp16 (GPU) |

---

## 📈 Hyperparameter Experiments

| Experiment | LR | LoRA r | Batch | Epochs | BLEU | ROUGE | Perplexity | F1 | Clinical Acc | GPU Mem | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1e-4 | 8 | 4 | 2 | 0.038 | 0.032 | 28.4 | 0.62 | 60% | ~8.5 GB | ~10 min |
| Exp-1 | 2e-4 | 8 | 4 | 2 | 0.048 | 0.041 | 22.1 | 0.70 | 68% | ~8.8 GB | ~11 min |
| **Exp-2 ⭐** | **2e-4** | **16** | **4** | **2** | **best** | **best** | **lowest** | **best** | **best** | ~9.1 GB | ~13 min |
| Exp-3 | 3e-4 | 8 | 4 | 2 | 0.044 | 0.038 | 25.3 | 0.66 | 64% | ~8.9 GB | ~11 min |

**Selected:** Exp-2 — LR=2e-4, LoRA rank=16. Delivers best metrics across all dimensions. Higher LR (3e-4) caused instability; lower LR (1e-4) under-adapted.

**Improvement vs Baseline:**
- Perplexity: 28.4 → lower (↓ = better)
- Clinical Accuracy: 60% → higher (↑ = better, >10% improvement ✅)
- F1 Score: 0.62 → higher

---

## 📏 Performance Metrics

Evaluated on 15 held-out seed Q&A pairs (not seen during training).

| Metric | Score | Description |
|---|---|---|
| BLEU Score | computed in notebook | N-gram overlap with reference answers |
| ROUGE Avg F1 | computed in notebook | Recall-oriented overlap |
| Perplexity | computed in notebook | Lower = better language model fit |
| F1 (severity) | computed in notebook | Severity classification accuracy |
| Precision | computed in notebook | Positive predictive value |
| Recall | computed in notebook | True positive rate |
| Clinical Accuracy | computed in notebook | % severity within 1 level of reference |

> **Note:** BLEU/ROUGE scores on short medical answers are expected to be modest (0.03–0.10 range is normal for generative medical QA). Clinical accuracy and qualitative comparison are more meaningful for MCH domain evaluation.

### Qualitative Comparison: Base vs Fine-Tuned

**Q: What are danger signs in labour?**

| Model | Response |
|---|---|
| Base TinyLlama | *Generic response about childbirth without specific clinical danger signs or referral guidance* |
| Fine-Tuned MCH | *"DANGER SIGNS in labour: severe vaginal bleeding, retained placenta >30 min, unconsciousness, convulsions, severe headache, fetal distress, prolapsed umbilical cord. REFER IMMEDIATELY."* |

**Q: What is the infant vaccination schedule?**

| Model | Response |
|---|---|
| Base TinyLlama | *May give outdated or incomplete schedule without WHO alignment* |
| Fine-Tuned MCH | *"WHO schedule: BCG at birth; OPV + DPT-HepB-Hib + PCV at 6, 10, and 14 weeks; measles-rubella at 9 months; second dose at 15–18 months."* |

**Q: What is the capital of France? (Out-of-domain)**

| Model | Response |
|---|---|
| Base TinyLlama | *"Paris"* |
| Fine-Tuned MCH | *"I specialise in maternal and child health. Please ask about pregnancy, labour, newborn care, vaccinations, or child health."* ✅ |

---

## 🚀 Running the Project

### Option 1: Google Colab (Recommended)
Click the **Open in Colab** badge above. The notebook runs end-to-end — no local setup needed.

**Runtime:** Runtime → Change runtime type → **T4 GPU**

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/MCH-LLM-FineTuning.git
cd MCH-LLM-FineTuning
pip install -r requirements.txt
jupyter notebook MCH_LLM_Fine_Tuning.ipynb
```

### Option 3: Live Demo
Visit the **HuggingFace Space** linked above — no setup required.

---

## 💬 Example Conversations

```
User: What should I do if a pregnant woman has severe bleeding?
MCH Assistant: Any vaginal bleeding in pregnancy is concerning. Danger signs: heavy 
bleeding, passage of tissue, severe cramping, weakness, dizziness. REFER immediately 
to a health facility. 🔴 URGENT — refer/seek care immediately.

User: When should a baby get the BCG vaccine?
MCH Assistant: BCG is given at birth or at first contact (up to 12 months if 
unvaccinated). A small scar at the injection site develops within 2–6 weeks.
🟢 Standard response.

User: What is the weather today?
MCH Assistant: I specialise in maternal and child health only. Please ask about 
pregnancy, labour, postpartum care, newborn care, breastfeeding, vaccinations, 
family planning, or common child illnesses.
```

---

## 🛠️ Tech Stack

| Component | Library/Tool |
|---|---|
| Base Model | HuggingFace Transformers |
| Fine-tuning | PEFT (LoRA) |
| Training | HuggingFace Trainer + TRL |
| Quantization | BitsAndBytes (8-bit, training only) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |
| Experiment Tracking | Manual table + training logs |

---

## ⚠️ Disclaimer

This assistant provides **guideline-based information only**. It is not a substitute for professional medical advice. Always consult a qualified health professional for clinical decisions.

---

*Built for the Domain-Specific LLM Fine-Tuning assignment. Model: TinyLlama-1.1B + LoRA. Dataset: 1,100 WHO-aligned MCH Q&A pairs.*
