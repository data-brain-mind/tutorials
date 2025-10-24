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
      name: Technische Universität München
  - name: Gilad Landau
    # url: ""
    affiliations:
      name: University of Oxford
  - name: Oiwi Parker Jones
    # url: ""
    affiliations:
      name: University of Oxford

# must be the exact same name as your blogpost
bibliography: 2025-11-24-neural-keyword-spotting-on-libribrain.bib

# Add a table of contents to your post.
toc:
  - name: Overview
  - name: Learning Goals
  - name: Outline
  - name: Notebook Access

# Below is an example of injecting additional post-specific styles.
# Delete this section if not needed.
# _styles: >
#   .your-custom-class {
#     /* your styles here */
#   }
---

## Overview

In the accompanying Jupyter Notebook tutorial for **Neural Keyword Spotting (KWS)** on the **LibriBrain** MEG dataset, you’ll run a compact temporal baseline, handle heavy class imbalance, and interpret **precision–recall** diagnostics (AUPRC, false alarms per hour at fixed recall).


## Learning Goals

- Frame KWS from continuous MEG as a rare-event detection problem.  
- Build and train a lightweight 1-D temporal model on consumer GPUs.  
- Use **AUPRC** and **FA/h at target recall** to reflect user-facing trade-offs.  
- Explore the first potentially useful application for non-invasive speech-BCIs!


## Outline
1. Data access & labeling — load MEG windows and keyword labels; inspect prevalence.  
2. Task formulation — windowing around events; train/val/test splits with safety checks.  
3. Model — compact temporal conv + attention pooling; focal-loss training.  
4. Evaluation — AUPRC, ROC (when defined), **FA/h @ target recall**; calibrated plots.  

## Notebook Access
- **Anonymous GitHub (Filebrowser)** [here](https://anonymous.4open.science/r/neurips-DBM-tutorial-8E79/Keyword_Detection_Tutorial.ipynb)
- **Direct Download** [here](https://anonymous.4open.science/api/repo/neurips-DBM-tutorial-8E79/file/Keyword_Detection_Tutorial.ipynb?v=4d18b5f6&download=true)