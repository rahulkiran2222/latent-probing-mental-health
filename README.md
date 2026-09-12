<div align="center">

# 🧠 Latent Probing for Mental Health Sentiment Classification

### Reading Mental-Health Signals from the Latent Space of Gemma 2 2B

<br>

<p>
  <img src="https://img.shields.io/badge/Model-Gemma%202%202B-8A2BE2?style=for-the-badge" alt="Gemma 2 2B">
  <img src="https://img.shields.io/badge/Layer-23-6A5ACD?style=for-the-badge" alt="Layer 23">
  <img src="https://img.shields.io/badge/Probe-Logistic%20Regression-4169E1?style=for-the-badge" alt="Logistic Regression">
  <img src="https://img.shields.io/badge/Accuracy-94.689%25-228B22?style=for-the-badge" alt="94.689 percent accuracy">
</p>

<p>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-444?style=flat-square">
  <img src="https://img.shields.io/badge/Pooling-Mean%20Pooling-444?style=flat-square">
  <img src="https://img.shields.io/badge/Max%20Tokens-64-444?style=flat-square">
  <img src="https://img.shields.io/badge/Training-Frozen%20LLM-444?style=flat-square">
</p>

<br>

**Can a simple linear probe recover mental-health distress information directly from the internal representations of a frozen language model?**

<br>

</div>

---

## ✦ Overview

This project investigates whether **mental-health distress signals are already encoded inside the latent representations of a pretrained large language model**.

Instead of fine-tuning Gemma 2 2B, the model is treated as a **frozen representation generator**.

Text is passed through the model, hidden states are extracted from a selected transformer layer, token representations are mean-pooled into fixed-size embeddings, and a lightweight linear classifier is trained on top.

The result:

> **94.689% local validation accuracy using only a Logistic Regression probe on Gemma 2 2B Layer 23 representations.**

This makes the project a study of **representation probing**, rather than conventional end-to-end fine-tuning.

---

## 🔬 The Core Research Question

<div align="center">

### What does Gemma already know internally?


                 ┌──────────────────────────┐
                 │        Input Text        │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Gemma 2 2B          │
                 │     Frozen Backbone      │
                 └────────────┬─────────────┘
                              │
                     Hidden State
                       Extraction
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        Layer 23          │
                 │      2304 dimensions     │
                 └────────────┬─────────────┘
                              │
                        Mean Pooling
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Latent Embedding     │
                 │        2304-D             │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Logistic Regression    │
                 │        C = 0.003         │
                 └────────────┬─────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │   0 / 1 Signal │
                     └────────────────┘
`

</div>

---

# 🧬 Why Latent Probing?

Large language models do not simply store information in their final output.

Their internal hidden states evolve through many transformer layers, progressively
building representations of linguistic, semantic, contextual, and task-relevant
information.

Latent probing asks a simple question:

> **If we freeze the model, can a small classifier decode a particular concept from its internal representations?**

This gives us a useful separation:


        Language Model
              │
              │ learns representations
              ▼
       ┌───────────────┐
       │ Latent Space  │
       └───────┬───────┘
               │
               │ probe
               ▼
       ┌───────────────┐
       │ Linear Model  │
       └───────┬───────┘
               │
               ▼
          Prediction


The language model is therefore **not being optimized for the downstream task**.

The probe is asking what information is already accessible.

---

# 🧠 Model

## Gemma 2 2B

The base model is:


google/gemma-2-2b


The model is used as a **frozen feature extractor**.

### No fine-tuning


Gemma parameters
      │
      ├── Frozen ✓
      │
      └── No gradient updates


Only the lightweight probe is trained.

---

# 🎯 Layer 23

The competition setup exposes 27 hidden-state layers:


0 ── 1 ── 2 ── 3 ── ... ── 22 ── [23] ── 24 ── 25 ── 26
                                  ▲
                                  │
                              TARGET LAYER


### Selected representation

<div align="center">

# Layer 23

**2,304-dimensional hidden representation**

</div>

Layer 23 was identified as the strongest layer for this task in the challenge
specification.

The probe therefore operates on the representation extracted from this layer.

---

# 🧮 Representation Extraction

For every input:

### 01 — Tokenize


Raw text
   ↓
Gemma tokenizer
   ↓
≤ 64 tokens


### 02 — Forward pass


Tokens
   ↓
Gemma 2 2B
   ↓
Hidden states


### 03 — Select Layer 23


Hidden states
   ↓
Layer 23
   ↓
Token × 2304 representation


### 04 — Mean pool


Token 1 ─┐
Token 2  │
Token 3  ├── Mean ──► 2304-D vector
Token 4  │
  ...    │
Token N ─┘


### 05 — Probe


2304-D latent vector
          ↓
Logistic Regression
          ↓
       0 / 1


---

# 📊 Dataset

The project uses:

**`ourafla/Mental-Health_Text-Classification_Dataset`**

The primary dataset contains four original categories:

| Original Category | Binary Target |
| :---------------: | :-----------: |
|       Normal      |      `0`      |
|     Depression    |      `1`      |
|      Suicidal     |      `1`      |
|      Anxiety      |      `1`      |

The resulting task is therefore:


                 ┌─────────────┐
                 │ Mental Text │
                 └──────┬──────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
         Normal                 Distress
           0                Depression
                            Suicidal
                             Anxiety


### Dataset size

**49,612 examples**

| Label      |    Samples |
| :--------- | ---------: |
| Normal     |     18,391 |
| Depression |     14,506 |
| Suicidal   |     11,212 |
| Anxiety    |      5,503 |
| **Total**  | **49,612** |

---

# ⚙️ Experimental Setup

| Parameter          | Configuration       |
| :----------------- | :------------------ |
| Base model         | `google/gemma-2-2b` |
| Model state        | Frozen              |
| Target layer       | **23**              |
| Hidden dimension   | **2,304**           |
| Maximum tokens     | **64**              |
| Pooling            | Mean pooling        |
| Train / validation | **80 / 20**         |
| Split              | Stratified          |
| Random seed        | **42**              |
| Probe              | Logistic Regression |
| Best `C`           | **0.003**           |
| Metric             | Accuracy            |

---

# 🏆 Results

<div align="center">

## 94.689%

### Best Local Validation Accuracy

<br>


████████████████████████████████████████████████████████████████████████████████████████████████░░░░░░░░░░
                                                                            94.689%


</div>

The strongest configuration was:

| Component           | Best Configuration      |
| :------------------ | :---------------------- |
| Model               | Gemma 2 2B              |
| Layer               | **23**                  |
| Pooling             | Mean                    |
| Probe               | **Logistic Regression** |
| `C`                 | **0.003**               |
| Validation Accuracy | **94.689%**             |

---

# ⚔️ Probe Comparison

Several lightweight probe configurations were evaluated.

| Probe                   | Configuration | Validation Accuracy |
| :---------------------- | :------------ | ------------------: |
| Linear SVM              | C = 0.3       |             93.742% |
| Linear SVM              | C = 1         |             93.722% |
| Linear SVM              | C = 3         |             93.712% |
| Linear SVM              | C = 10        |             93.752% |
| Logistic Regression     | C = 0.001     |             94.427% |
| **Logistic Regression** | **C = 0.003** |         **94.689%** |

### Best configuration


                 Validation Accuracy

Linear SVM       ███████████████████████████████████████████████  ~93.75%

Logistic Reg.    █████████████████████████████████████████████████  94.689%


The result suggests that the representation itself provides a strong signal,
while probe selection still affects the final separability achieved by the
classifier.

---

# 🔎 What Does the Result Tell Us?

## 01 — The signal is already present

A 94.689% validation accuracy from a lightweight linear classifier suggests
that substantial task-relevant information can be recovered from the frozen
latent representation.

---

## 02 — Fine-tuning was not required

The underlying Gemma model was not fine-tuned for this classification task.

The workflow is:


                 PRETRAINED MODEL
                       │
                       ▼
                Frozen Gemma 2B
                       │
                       ▼
                  Layer 23
                       │
                       ▼
               Latent Embedding
                       │
                       ▼
               Linear Classifier
                       │
                       ▼
                    Signal


This makes the experiment particularly useful for studying **what is already
encoded inside the model**.

---

## 03 — Probe simplicity is intentional

The final classifier is deliberately lightweight.

There is no large neural classification head.

There is no task-specific fine-tuning.

There is no additional deep architecture.

The experiment instead asks:

> **How much can a simple linear boundary recover from the model's latent space?**

---

# 🧭 Representation → Prediction

<div align="center">


┌───────────────────────────────────────────────────────────┐
│                        INPUT TEXT                         │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                       GEMMA 2 2B                          │
│                       FROZEN                              │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                        LAYER 23                           │
│                       2304-D                              │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                      MEAN POOLING                         │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                    LATENT REPRESENTATION                  │
│                         2304-D                            │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                LOGISTIC REGRESSION                        │
│                       C = 0.003                            │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   0 = Normal     │
                    │   1 = Distress   │
                    └──────────────────┘


</div>

---

# 💾 Representation Caching

A major practical advantage of this workflow is that the expensive language
model computation only needs to happen once.


                Gemma 2 2B
                     │
                     ▼
              Hidden States
                     │
                     ▼
              ┌──────────────┐
              │ Cache .npz   │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
        SVM       Logistic    Future
       Probe       Probe      Probes


This allows multiple probe configurations to be evaluated without repeatedly
running Gemma over the entire dataset.

---

# 🧪 Why This Is Interesting

This project is not primarily about building another text classifier.

The deeper objective is to investigate **latent information accessibility**.

A high-performing classifier operating on frozen representations provides
evidence that the representation contains information that is linearly
recoverable for the target task.

That creates a useful experimental separation:


             MODEL KNOWLEDGE
                    │
                    ▼
             LATENT SPACE
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       Accessible          Difficult to
        signal             decode
          │
          ▼
        PROBE


---

# 🧩 Repository Structure


latent-probing-mental-health/
│
├── README.md
├── train_probe.py
├── classifier.py
├── trained_probe.joblib
├── requirements.txt
├── results/
│   └── ...
└── LICENSE


### Key files

| File                   | Purpose                                  |
| :--------------------- | :--------------------------------------- |
| `train_probe.py`       | Extract representations and train probes |
| `classifier.py`        | CodaBench inference interface            |
| `trained_probe.joblib` | Final trained probe                      |
| `results/`             | Experimental results                     |
| `requirements.txt`     | Python dependencies                      |
| `README.md`            | Project documentation                    |

---

# 🚀 Reproducibility

The primary configuration can be reproduced with:


Model:
google/gemma-2-2b

Layer:
23

Maximum sequence length:
64

Pooling:
Mean

Split:
80/20 stratified

Seed:
42

Probe:
Logistic Regression

C:
0.003


The expensive hidden-state extraction is separated from probe training,
allowing the cached embeddings to be reused for further experiments.

---

# 🏁 CodaBench

This project was developed for the:

**Latent Probing for Mental Health Sentiment Classification**

challenge.

The competition requires a classifier compatible with the provided
inference interface.

The final submission uses:


classifier.py
        +
trained_probe.joblib


The inference classifier loads the trained probe and applies it to the
provided latent representations.

---

# ⚠️ Limitations

This project is a machine-learning representation-probing experiment.

It is **not a clinical diagnostic system**.

The classifier learns patterns associated with the dataset labels and should
not be interpreted as a medical diagnosis or mental-health assessment tool.

Performance may depend on:

* Dataset composition
* Dataset labeling
* Distribution shift
* Representation choice
* Transformer layer
* Pooling strategy
* Probe architecture

The reported validation score therefore measures benchmark performance rather
than clinical validity.

---

# 🔬 Research Series

This project is part of a broader exploration of **latent probing in
large language models**.

### Project 01

**Latent Probing for Toxicity Detection**

Explores whether toxicity-related information can be recovered from Gemma 2 2B
hidden representations.

### Project 02

**Latent Probing for Mental Health Sentiment Classification**

Explores whether mental-health distress information can similarly be decoded
from the model's latent space.


┌─────────────────────────────┐
│       GEMMA 2 2B            │
└──────────────┬──────────────┘
               │
        Hidden Representations
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   PROJECT 01        PROJECT 02
   Toxicity          Mental Health
       │                │
       ▼                ▼
     Probe            Probe
       │                │
       └───────┬────────┘
               ▼
        LATENT PROBING
        RESEARCH THREAD


The broader research question is:

> **What information is encoded in pretrained language-model representations,
> and how much of that information can be recovered using simple probes?**

---

# 📌 Key Takeaways

<div align="center">

| Finding                   |    Result   |
| :------------------------ | :---------: |
| Frozen Gemma 2B           |      ✓      |
| Layer 23 representation   |      ✓      |
| Mean-pooled latent vector |      ✓      |
| Lightweight linear probe  |      ✓      |
| No LLM fine-tuning        |      ✓      |
| Best local validation     | **94.689%** |

</div>

---

## 📈 Final Result

<div align="center">

# 94.689%

### Validation Accuracy

**Gemma 2 2B · Layer 23 · Mean Pooling · Logistic Regression**

<br>

`C = 0.003`

<br>

---

### From Text → Latent Space → Recoverable Signal

</div>

---

## 📜 License

This project is released under the terms of the accompanying license.

---

<div align="center">

### Built as part of a continuing investigation into

### **LLM representations, latent probing, and model interpretability.**

<br>

⭐ **If you find this research direction interesting, consider exploring the other projects in the series.**

</div>

