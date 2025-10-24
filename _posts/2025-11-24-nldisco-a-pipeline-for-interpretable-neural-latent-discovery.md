---
layout: distill
title: "NLDisco: A Pipeline for Interpretable Neural Latent Discovery"
description: 
date: 2025-11-24
future: true
htmlwidgets: true

# Camera-ready version deployed from OpenReview
# Set to true to hide until authors complete their updates via PR
hidden: false

authors:
  - name: Anaya Gaelle Pouget
    # url: ""
    affiliations:
      name: University College London, University of London
  - name: Jai Bhagat
    url: "https://jkbhagatio.io"
    affiliations:
      name: University College London, University of London
  - name: Sara Molas-Medina
    # url: ""
    affiliations:
      name: independent

# must be the exact same name as your blogpost
bibliography: 2025-11-24-nldisco-a-pipeline-for-interpretable-neural-latent-discovery.bib

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

This tutorial introduces **NLDisco** (**N**eural **L**atent **Disco**very pipeline), a machine learning approach for discovering interpretable features in high-dimensional neural recordings. The method leverages sparse autoencoders (SAEs) to identify meaningful latent dimensions that correspond to specific behavioural or environmental variables.

**What you’ll accomplish:**
- Train sparse autoencoders on neural spike data to automatically discover interpretable features
- Use Matryoshka architecture to capture both broad and specific neural patterns simultaneously
- Validate discoveries through an interactive dashboard that visualizes feature-behavior relationships