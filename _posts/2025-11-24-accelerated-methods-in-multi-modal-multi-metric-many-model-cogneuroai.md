---
layout: distill
title: "Accelerated Methods in {Multi-Modal, Multi-Metric, Many-Model} CogNeuroAI"
description: A tutorial showcasing a number of (GPU-accelerated) methods for probing the representational alignment of brains, minds, and machines.
date: 2025-11-24
future: true
htmlwidgets: true

# Camera-ready version deployed from OpenReview
# Set to true to hide until authors complete their updates via PR
hidden: false

authors:
  - name: Colin Conwell
    # url: ""
    affiliations:
      name: Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2025-11-24-accelerated-methods-in-multi-modal-multi-metric-many-model-cogneuroai.bib

# Add a table of contents to your post.
toc:
  - name: Introduction

# Below is an example of injecting additional post-specific styles.
# Delete this section if not needed.
# _styles: >
#   .your-custom-class {
#     /* your styles here */
#   }
---

In this tutorial, we cover a series of modern computational modeling methods in CogNeuroAI (the emergent, interdisciplinary intersection of Cognitive Science, Neuroscience, and AI), with a particular focus on accelerating research in representational alignment.

The tutorial is written end-to-end as an interactive Jupyter notebook, and is designed not only to provide a comprehensive and generally accessible methodological overview to audience members of diverse backgrounds, but also to provide access to highly-optimized software that will allow participants to quickly and readily adapt the tutorial's methods to their own research.

The tutorial is subdivided in multiple related, but otherwise containerized parts, and includes:
- Introduction to DeepJuice: A custom, highly-optimized, end-to-end GPU-accelerated library designed to facilitate high-throughput brain & behavioral modeling at scale
- The step-by-step reproduction of a recent large-scale, controlled model comparison pipeline on the 7T fMRI Natural Scenes Dataset
- A case study of cross-modal representational alignment (using language models to predict image-evoked brain activity in human vental visual cortex)
- A demonstration of hypothesis-driven, vector-semantic mapping: a technique (derived from relative representation / anchor-point embedding analysis) that allows researchers to query the underlying structure of representational alignment with natural language.

The tutorial code is available at the following anonymous GitHub link: https://anonymous.4open.science/r/DBM-Tutorial-CFC4/DBM2025.ipynb