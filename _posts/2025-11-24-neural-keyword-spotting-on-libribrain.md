---
layout: distill
title: "Neural Keyword Spotting on LibriBrain"
description: "End-to-end walkthrough for Neural Keyword Spotting (KWS) on the LibriBrain MEG corpus: load data, frame the task, train a compact baseline, and evaluate with precision–recall metrics tailored to extreme class imbalance."
date: 2025-11-24
future: true
htmlwidgets: true

# Camera-ready version deployed from OpenReview
# Set to true to hide until authors complete their updates via PR
hidden: false

authors:
  - name: Gereon Elvers
    url: "https://gereonelvers.com"
    affiliations:
      name: PNPL, University of Oxford
  - name: Gilad Landau
    affiliations:
      name: PNPL, University of Oxford
  - name: Oiwi Parker Jones
    affiliations:
      name: PNPL, University of Oxford

# must be the exact same name as your blogpost
bibliography: 2025-11-24-neural-keyword-spotting-on-libribrain.bib

# Add a table of contents to your post.
toc:
  - name: Introduction
  - name: Motivation and Context
  - name: Dataset and Methodology
    subsections:
      - name: The LibriBrain Dataset
      - name: Task Formulation
      - name: Model Architecture
      - name: Training Strategy
  - name: Evaluation Metrics
  - name: Computational Requirements
  - name: Learning Goals
  - name: Tutorial Structure
  - name: Notebook Access

# Below is an example of injecting additional post-specific styles.
# Delete this section if not needed.
# _styles: >
#   .your-custom-class {
#     /* your styles here */
#   }
---

## Introduction

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-11-24-neural-keyword-spotting-on-libribrain/task-graphic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of the neural keyword spotting task: A participant listens to audiobook speech while MEG sensors record brain activity. The system processes neural signals to detect when specific keywords (like "Watson") are heard, producing a probability score for each word window.
</div>

> **Note**: This tutorial is released in conjunction with our DBM workshop paper *"Elementary, My Dear Watson: Non-Invasive Neural Keyword Spotting in the LibriBrain Dataset"*<d-cite key="elvers2025elementary"></d-cite>. The tutorial provides a comprehensive introduction as well as a hands-on, pedagogical walkthrough of the methods and concepts presented in the paper.

**Neural Keyword Spotting (KWS)** from brain signals presents a promising direction for non-invasive brain–computer interfaces (BCIs), with potential applications in assistive communication technologies for individuals with speech impairments. While invasive BCIs have achieved remarkable success in speech decoding<d-cite key="willett2023speech,metzger2023neuroprosthesis"></d-cite>, non-invasive approaches using magnetoencephalography (MEG) or electroencephalography (EEG) remain challenging due to lower signal-to-noise ratios and the difficulty of detecting brief, rare events in continuous neural recordings.

This tutorial demonstrates how to build and evaluate a neural keyword spotting system using the **LibriBrain dataset**<d-cite key="ozdogan2025libribrain"></d-cite>—a large-scale MEG corpus with over 50 hours of recordings from a single participant listening to audiobooks. We focus on the practical challenges of extreme class imbalance, appropriate evaluation metrics, and techniques for training models that can distinguish keyword occurrences from the continuous stream of speech.

## Motivation and Context

### Why Keyword Spotting?

Full speech decoding from non-invasive brain signals remains an open challenge. However, **keyword spotting**—detecting specific words of interest—offers a more tractable goal that could still enable meaningful communication. Even detecting a single keyword reliably (a "1-bit channel") could significantly improve quality of life for individuals with severe communication disabilities, allowing them to:

- Answer yes/no questions
- Signal alerts or requests
- Control devices through specific command words
- Maintain basic communication when other channels fail

### The Challenge: Rare Events in Noisy Data

Keyword spotting from MEG presents two fundamental challenges:

1. **Extreme Class Imbalance**: Even short, common words like "the" represent only ~5.5% of all words in naturalistic speech. Target keywords like "Watson" appear in just 0.12% of word windows, creating a severe imbalance.

2. **Low Signal-to-Noise Ratio**: Unlike invasive recordings with electrode arrays placed directly on the cortex, non-invasive MEG/EEG sensors sit outside the skull, capturing attenuated and spatially blurred neural signals mixed with physiological and environmental noise.

These challenges require specialized techniques, which we cover in this tutorial.

## Dataset and Methodology

### The LibriBrain Dataset

The **LibriBrain dataset**<d-cite key="ozdogan2025libribrain"></d-cite> is a publicly available MEG corpus featuring over 50 hours of continuous recordings from a single participant listening to Sherlock Holmes audiobooks. The dataset is released as a set of preprocessed HDF5 files with word- and phoneme-level event annotation for each session, collected using a MEGIN Triux™ Neo system. The dimension of the currently released data is 306 sensor channels x 250 Hz.

### Task Formulation

We frame keyword detection as **event-referenced binary classification**:

- **Input**: MEG signals (306 channels × T timepoints) windowed around word onsets
- **Output**: Probability p ∈ [0, 1] that the target keyword occurs in this window
- **Window length**: Keyword duration + buffers (pre-/post-onset)

This differs from continuous detection by:
1. Focusing on word boundaries (where linguistic information peaks)
2. Avoiding the combinatorial explosion of sliding windows
3. Leveraging precise temporal alignment from annotations

**Data Splits**: We use multiple training sessions and dynamically select validation/test sessions based on keyword prevalence to ensure sufficient positive examples in held-out sets.

### Model Architecture

The tutorials baseline model addresses the challenges through three components:

> **Note**: The notebook first demonstrates individual components with simplified examples (e.g., `ConvTrunk` with stride-2), then presents the full training architecture below.

#### 1. Convolutional Trunk
The model begins with a Conv1D layer projecting the 306 MEG channels to 128 dimensions, followed by a residual block<d-cite key="he2016deep"></d-cite>. A key design choice is **aggressive temporal downsampling**: a stride-25 convolution with kernel size 50 reduces the sequence length by ~25× while expanding the receptive field. Two additional Conv1D layers refine the 128-dimensional representation.

```python
self.trunk = nn.Sequential(
    nn.Conv1d(306, 128, 7, 1, padding='same'),
    ResNetBlock1D(128),
    nn.ELU(),
    nn.Conv1d(128, 128, 50, 25, 0),  # stride-25 downsampling
    nn.ELU(),
    nn.Conv1d(128, 128, 7, 1, padding='same'),
    nn.ELU(),
)
```

#### 2. Temporal Attention
The trunk output is projected to 512 dimensions before splitting into two parallel 1×1 convolution heads: one producing per-timepoint logits, the other producing attention scores. The attention mechanism<d-cite key="ilse2018attention"></d-cite> learns to focus on brief, informative time windows (e.g., around keyword onsets) while down-weighting noise.

```python
self.head = nn.Sequential(nn.Conv1d(128, 512, 4, 1, 0), nn.ReLU(), nn.Dropout(0.5))
self.logits_t = nn.Conv1d(512, 1, 1, 1, 0)
self.attn_t = nn.Conv1d(512, 1, 1, 1, 0)

def forward(self, x):
    h = self.head(self.trunk(x))
    logit_t = self.logits_t(h)
    attn = torch.softmax(self.attn_t(h), dim=-1)
    return (logit_t * attn).sum(dim=-1).squeeze(1)
```

#### 3. Loss Functions for Extreme Imbalance
Standard cross-entropy fails under extreme class imbalance. We employ two complementary losses:

- **Focal Loss**<d-cite key="lin2017focal"></d-cite>: Down-weights easy negatives by $(1-p_t)^\gamma$, with class prior $\alpha=0.95$ matching the <1% base rate. This prevents "always negative" collapse.

- **Pairwise Ranking Loss**<d-cite key="burges2010ranknet"></d-cite>: Directly optimizes the ordering of positive vs. negative scores, improving precision-recall trade-offs:

```python
def pairwise_logistic_loss(scores, targets):
    pos_idx = (targets == 1).nonzero()
    neg_idx = (targets == 0).nonzero()
    # Sample pairs and penalize inversions
    margins = scores[pos_idx] - scores[sampled_neg_idx]
    return torch.log1p(torch.exp(-margins)).mean()
```

### Training Strategy

**Balanced Sampling**: We construct training batches with ~10% positive rate (vs. natural <1%) by:
1. Including most/all positive examples
2. Subsampling negatives proportionally
3. Shuffling each batch

This ensures gradients aren't starved by all-negative batches while keeping evaluation on natural class priors for realistic metrics.

**Preprocessing**: The dataset applies per-channel z-score normalization and clips outliers beyond ±10σ before feeding data to the model.

**Data Augmentation**<d-cite key="buda2018systematic"></d-cite> (applied during training only):
- Temporal shifts: randomly roll each sample by ±4 timepoints (±16ms at 250 Hz)
- Additive Gaussian noise: σ=0.01 added to normalized signals

**Regularization**: Dropout (p=0.5), weight decay<d-cite key="loshchilov2019decoupled"></d-cite> (1e-4), and early stopping on validation loss.

## Evaluation Metrics

Traditional accuracy is meaningless under extreme imbalance (always predicting "no keyword" yields >99% accuracy). We employ metrics that reflect real-world BCI deployment:

### Threshold-Free Metrics

**Area Under Precision-Recall Curve (AUPRC)**<d-cite key="saito2015precision,davis2006relationship"></d-cite>:
- Baseline equals positive class prevalence (~0.001 for "Watson" on the full dataset, ~0.005 on the test set chosen to maximize prevalence)
- Aim for 2–10× improvement over chance
- More informative than AUROC under heavy imbalance

**Precision-Recall Trade-off**:
- **Precision**: Fraction of predicted keywords that are correct (controls false alarms)
- **Recall**: Fraction of true keywords detected

### User-Facing Deployment Metrics

**False Alarms per Hour (FA/h)**:
- Practical constraint: target <10 FA/h for usability
- Computed as: `(False Positives / total_seconds) × 3600`
- Evaluated at fixed recall targets (e.g., 0.2, 0.4, 0.6)

**Operating Point Selection**:
Choose threshold on validation to meet FA/h or precision targets; report test results at that fixed threshold.

### Performance Interpretation

- **Chance**: Prevalence (% of words matching the keyword)
- **2–5× Chance**: Modest but meaningful improvement
- **>10× Chance**: Strong performance for this challenging task

## Computational Requirements
- **GPU**: Google Colab free tier (T4/L4 GPU) sufficient
- **Training Time**: ~30 minutes for the baseline on default configuration
- **Memory**: <16 GB GPU RAM with batch size 64
- **Dataset**: Automatically downloaded by the `pnpl` library (~50 GB for the full set, ~5GB for the default subset)

The tutorial is designed to run on consumer hardware by training on a subset of data. To scale to the full 50+ hours of data, increase training sessions in the configuration and use a higher-tier GPU (V100/A100).

## Learning Goals

By working through this tutorial, you will:

1. **Frame KWS from continuous MEG** as a rare-event detection problem with event-referenced windowing
2. **Handle extreme class imbalance** through balanced sampling, focal loss, and pairwise ranking
3. **Build a lightweight temporal model** (Conv1D + attention) trainable on consumer GPUs
4. **Evaluate with appropriate metrics**: AUPRC, FA/h at fixed recall, precision-recall curves
5. **Understand trade-offs** between sensitivity (recall), false alarm rate, and practical usability
6. **Gain hands-on experience** with a real-world non-invasive BCI dataset

## Tutorial Structure

The accompanying Jupyter notebook provides a complete, executable walkthrough:

1. **Setup & Configuration** — Install dependencies, configure paths and hyperparameters
2. **Dataset Exploration** — Inspect HDF5 files (MEG signals) and TSV files (annotations)
3. **Problem Formulation** — Visualize challenges (class imbalance, signal noise)
4. **Model Components** — Interactive demos of each architectural component:
   - Convolutional trunk (spatial-temporal processing)
   - Temporal attention (adaptive pooling)
   - Focal loss (imbalance handling)
   - Pairwise ranking (order-based training)
   - Balanced sampling (batch composition)
5. **Training** — Full training loop with PyTorch Lightning, early stopping, logging
6. **Evaluation** — AUPRC, ROC, FA/h curves, confusion matrices, threshold analysis
7. **Next Steps** — Suggested experiments (different keywords, architectures, augmentations)


## Notebook Access

Access the full interactive tutorial:

<div style="display: flex; gap: 10px; margin: 20px 0;">
  <a href="https://colab.research.google.com/github/neural-processing-lab/libribrain-keyword-experiments/blob/main/tutorial/Keyword_Spotting_Tutorial.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/neural-processing-lab/libribrain-keyword-experiments/blob/main/tutorial/Keyword_Spotting_Tutorial.ipynb" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-View%20Source-blue?logo=github" alt="View on GitHub"/>
  </a>
</div>

**Links**:
- **Interactive (Colab)**: [Open in Google Colab](https://colab.research.google.com/github/neural-processing-lab/libribrain-keyword-experiments/blob/main/tutorial/Keyword_Spotting_Tutorial.ipynb)
- **Source (GitHub)**: [View on GitHub](https://github.com/neural-processing-lab/libribrain-keyword-experiments/blob/main/tutorial/Keyword_Spotting_Tutorial.ipynb)
- **Workshop Paper**: [arXiv:2510.21038](https://arxiv.org/abs/2510.21038)
- **LibriBrain Dataset**: [View on HuggingFace](https://huggingface.co/datasets/pnpl/LibriBrain)

**Requirements**: A Google account for Colab, or local Jupyter Notebook install with Python 3.10+

---

Besides the accompanying workshop paper <d-cite key="elvers2025elementary"></d-cite>, this tutorial builds on work from the 2025 LibriBrain Competition<d-cite key="landau2025pnpl"></d-cite> centered around the LibriBrain dataset<d-cite key="ozdogan2025libribrain"></d-cite>. These papers contain more comprehensive bibliographies which might be helpful for readers seeking additional context.