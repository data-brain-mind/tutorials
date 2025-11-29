---
layout: distill
title: An Overview of the Neuropixels Visual Behavior Dataset From the Allen Institute
description: Tutorial on analyzing the Visual Behavior Neuropixels dataset from the Allen Institute
date: 2025-11-24
future: true
htmlwidgets: true

# Camera-ready version deployed from OpenReview
# Set to true to hide until authors complete their updates via PR
hidden: false

authors:
  - name: Corbett Bennett
    url: "https://alleninstitute.org/person/corbett-bennett/"
    affiliations:
      name: Allen Institute
  - name: Su-Yee J Lee
    # url: ""
    affiliations:
      name: Allen Institute
  - name: Josh Siegle
    # url: ""
    affiliations:
      name: Allen Institute
  - name: Marina Garrett
    # url: ""
    affiliations:
      name: Allen Institute
  - name: Saskia EJ de Vries
    # url: ""
    affiliations:
      name: Allen Institute
  - name: Shawn R Olsen
    url: "https://scholar.google.com/citations?user=huDkgmYAAAAJ&hl=en"
    affiliations:
      name: Allen Institute

# must be the exact same name as your blogpost
bibliography: 2025-11-24-an-overview-of-the-neuropixels-visual-behavior-dataset-from-the-allen-institute.bib

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


The following tutorial will provide a brief introduction to this dataset then demonstrate how to load an example session and do some simple analysis. We'll cover the following topics:

* Introduction to the experiment
* Accessing the dataset
* Reading the project metadata
* Loading an experiment for analysis
* Plotting neural activity aligned to stimuli
* Using optotagging to infer cell type
* Accessing LFP data
* Analyzing behavioral data during the visual change detection task


## Introduction to the experiment

Our ability to perceive the sensory environment and flexibly interact with the world requires the coordinated action of neuronal populations distributed throughout the brain. To further our understanding of the neural basis of behavior, the Visual Behavior project leveraged the Allen Brain Observatory pipeline (diagrammed below; gray panels refer to the companion [Visual Behavior Optical Physiology dataset](https://portal.brain-map.org/circuits-behavior/visual-behavior-2p)) to collect a large-scale, highly standardized dataset consisting of recordings of neural activity in mice that have learned to perform a visual change detection task. This dataset can be used to investigate how patterns of spiking activity across the visual cortex and thalamus are related to behavior and also how these activity dynamics are influenced by task-engagement and prior visual experience.  

The Visual Behavior Neuropixels dataset includes 153 sessions from 81 mice. These data are made openly accessible, with all recorded timeseries, behavioral events, and experimental metadata conveniently packaged in Neurodata Without Borders (NWB) files that can be accessed and analyzed using our open Python software package, the  [AllenSDK](https://github.com/AllenInstitute/AllenSDK).

{% include figure.html path="assets/img/2025-11-24-an-overview-of-the-neuropixels-visual-behavior-dataset-from-the-allen-institute/pipeline.png" class="img-fluid" style="width:800px;" %}

### The visual change detection task

The Visual Behavior Optical Physiology and Visual Behavior Neuropixels projects are built upon a change detection behavioral task. Briefly, in this go/no-go task, mice are shown a continuous series of briefly presented visual images and they earn water rewards by correctly reporting when the identity of the image changes (diagrammed below). Five percent of images are omitted, allowing for analysis of expectation signals. 

{% include figure.html path="assets/img/2025-11-24-an-overview-of-the-neuropixels-visual-behavior-dataset-from-the-allen-institute/change_detection_task.png" class="img-fluid" style="width:900px;" %}

### Neuropixels recordings throughout visual cortex and thalamus

This dataset includes multi-regional Neuropixels recordings from up to 6 probes at once. The probes target six visual cortical areas including VISp, VISl, VISal, VISrl, VISam, and VISpm. In addition, multiple subcortical areas are also typically measured, including visual thalamic areas LGd and LP as well as units in the hippocampus and midbrain. In addition to spiking activity, NWB files contain local field potential (LFP) data from each probe insertion.

Recordings were made in three genotypes: C57BL6J, Sst-IRES-Cre; Ai32, and Vip-IRES-Cre; Ai32. By crossing Sst and Vip lines to the Ai32 ChR2 reporter mouse, we were able to activate putative Sst+ and Vip+ cortical interneurons by stimulating the cortical surface with blue light during an optotagging protocol at the end of each session.

{% include figure.html path="assets/img/2025-11-24-an-overview-of-the-neuropixels-visual-behavior-dataset-from-the-allen-institute/probe_diagram.png" class="img-fluid" style="width:600px;" %}

### Investigating the impact of stimulus novelty on neural responses and behavior

To allow analysis of stimulus novelty on neural responses and behavior, two different images sets were used in the recording sessions: G and H (diagrammed below). Both image sets comprised 8 natural images. Two images were shared across the two image sets (purple in diagram), enabling within session analysis of novelty effects. Mice took one of the following three trajectories through training and the two days of recording:

1) Train on G; see G on the first recording day; see H on the second recording day

2) Train on G; see H on the first recording day; see G on the second recording day

3) Train on H; see H on the first recording day; see G on the second recording day

The numbers in the *Training and Recording Workflow* bubble below give the total recording sessions of each type in the dataset.


{% include figure.html path="assets/img/2025-11-24-an-overview-of-the-neuropixels-visual-behavior-dataset-from-the-allen-institute/recording_strategy.png" class="img-fluid" style="width:600px;" %}


#### [Tutorial link](https://github.com/AllenNeuralDynamics/neurips-vbn-tutorial/blob/main/code/vbn_neurips_tutorial.ipynb)

### Resources 

#### Data Resources associated with this dataset 

| Data Resource    | Format | Uniform Resource ID (URI) |
| -------- | ------- | ------- |
| Behavior + spike data | NWB | s3://visual-behavior-neuropixels-data/visual-behavior-neuropixels/, https://doi.org/10.48324/dandi.000713/0.240702.1725 |
| Behavior training data | NWB | s3://visual-behavior-neuropixels-data/visual-behavior-neuropixels/, https://doi.org/10.48324/dandi.000713/0.240702.1725 |
| Local Field Potential (LFP) data | NWB| s3://visual-behavior-neuropixels-data/visual-behavior-neuropixels/, https://doi.org/10.48324/dandi.000713/0.240702.1725 |
| Raw AP band data | Binary | s3://allen-brain-observatory/visual-behavior-neuropixels/raw-data/ |
| Raw LFP band data | Binary | s3://allen-brain-observatory/visual-behavior-neuropixels/raw-data/ |
| Eye, face, body videos  | MP4  | s3://allen-brain-observatory/visual-behavior-neuropixels/raw-data/ |

<br>

#### Technical Resources associated with this dataset 

| Technical Resource    | Format | Uniform Resource ID (URI) |
| -------- | ------- | ------- |
| AllenSDK tutorials| Jupyter Notebooks | https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html|
| Visual Behavior Neuropixels Databook | Jupyter Book | https://allenswdb.github.io/physiology/ephys/visual-behavior/VB-Neuropixels.html |
| Visual Behavior Neuropixels Technical Whitepaper | PDF| https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/f7/06/f706855a-a3a1-4a3a-a6b0-3502ad64680f/visualbehaviorneuropixels_technicalwhitepaper.pdf|

*AllenSDK tutorials:* Jupyter notebooks demonstrating how to access data with the AllenSDK and perform basic analysis on behavior, spike and LFP data. 

*Visual Behavior Neuropixels Databook:* Jupyter Book providing background information about the experimental design and change detection task, as well as in-depth descriptions of all metadata tables in the dataset. 

*Visual Behavior Neuropixels Technical Whitepaper:* Document providing detailed information about the Allen Brain Observatory data collection pipeline and rig hardware. 

In addition to these materials, please refer to the following manuscript, which provides a basic characterization of the dataset and which we intend as the primary citation for those using this resource: https://www.biorxiv.org/content/10.1101/2025.10.17.683190v1. 

