---
layout: distill
title: "Reading Observed At Mindless Moments (ROAMM): A Simultaneous EEG and Eye-Tracking Dataset of Natural Reading with Attention Annotations"
description: "ROAMM (Reading Observed At Mindless Moments) is a large-scale multimodal dataset of EEG and eye-tracking during naturalistic reading, with precise annotations of mind-wandering episodes to advance research in attention, language, and machine learning."
date: 2025-09-05
future: true
htmlwidgets: true
hidden: false

authors:
  - name: "Haorui Sun"
    affiliations:
      - name: "University of Vermont"
  - name: "Ardyn Vivienne Olszko"
    affiliations:
      - name: "University of Vermont"
  - name: "Yida Chen"
    affiliations:
      - name: "Harvard University"
  - name: "Hazen Kellner"
    affiliations:
      - name: "University of Vermont"
  - name: "Niharika Singh"
    affiliations:
      - name: "University of Vermont"
  - name: "Lincoln Lewisequerre"
    affiliations:
      - name: "University of Vermont"
  - name: "David C. Jangraw"
    affiliations:
      - name: "University of Vermont"

bibliography: "2025-11-24-reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations.bib"

toc:
  - name: "Introduction and motivation"
    subsections:
      - name: "Neural decoding models: from science fiction to reality"
      - name: "Simultaneous EEG and eye-tracking for cognitive dataset"
      - name: "What are some challenges?"
        subsections:
          - name: "The rarity of datasets on natural reading"
          - name: "Frequent mind-wandering (off-task thoughts) during reading"
      - name: "Mind-wandering is sneaky, so how do we study it?"
        subsections:
          - name: "Previous approaches and their drawbacks"
          - name: "Our novel ReMind paradigm"
      - name: "Introducing the ROAMM dataset"

  - name: "ROAMM dataset"
    subsections:
      - name: "Participants"
      - name: "The ReMind paradigm"
      - name: "Data acquisition and preprocessing"
        subsections:
          - name: "Eye-tracking"
          - name: "EEG"
      - name: "Dataset format"
      - name: "Dataset scale"
      - name: "Data validation"
        subsections:
          - name: "EEG and eye-tracking recording quality"
          - name: "EEG and eye-tracking alignment"
          - name: "Fixation-to-word mapping"
          - name: "Reliable MW onset"
      - name: "Data accessibility and availability"
      - name: "How to use ROAMM"

  - name: "Open questions for ML practitioners using ROAMM"
    subsections:
      - name: "Learn shared representation from EEG and eye-tracking"
      - name: "Build a momentary human attention decoder"
      - name: "Is human attention what we need for neural decoding?"
      - name: "Use EEG, eye-tracking, human attention, and reading text to predict comprehension"

  - name: "Conclusions"
---


## TL;DR
**ROAMM (Reading Observed At Mindless Moments) is a large-scale multimodal dataset of EEG and eye-tracking data during naturalistic reading, with precise annotations of mind-wandering episodes to advance research in attention, language, and machine learning.**

## 1. Introduction and motivation 
### 1.1 Neural decoding models: from science fiction to reality
Many of us have seen movies or read stories where people can “mind read,” such as Professor X, an exceptionally powerful telepath who can read and control the thoughts of others. While this belongs to science fiction, advances in technology and computation are bringing us closer to decoding aspects of human thought in real life. Neural decoding models powered by modern natural language processing and large language models have begun to approximate how humans process and generate language. These models, in turn, have helped researchers to understand how human brains accomplish the same tasks. 
### 1.2 Simultaneous EEG and eye-tracking for cognitive dataset
Building and refining powerful neural decoding models require large-scale and high-quality cognitive data. A powerful way to capture such cognitive data is to combine eye-tracking with electroencephalography (EEG). Eye-tracking captures gaze positions, which link visual input to specific task stimuli or words at given time. EEG, meanwhile, provides non-invasive and relatively low-cost recordings of brain dynamics with high temporal resolution. 
### 1.3 What are some challenges?
#### 1.3.1 The rarity of datasets on natural reading
Most existing simultaneous EEG and eye-tracking datasets focus on non-linguistic tasks. For example, EEGEyeNet provides large-scale recordings of eye movements while participants view simple visual symbols <d-cite key="Kastrati2021"></d-cite>, and EEGET-RSOD <d-cite key="He2025"></d-cite> records brain and eye activity as participants search for targets in remote sensing images. For reading specifically, the most well-known datasets are ZuCo <d-cite key="Hollenstein2018"></d-cite> and ZuCo 2.0 <d-cite key="Hollenstein2019"></d-cite>, which combine EEG and eye-tracking during sentence-level reading (i.e., each stimulus is a single sentence). These reading resources have become popular and invaluable for advancing EEG-to-text decoding and related computational models. 
#### 1.3.2 Frequent mind-wandering (off-task thoughts) during reading 
You might argue that existing datasets could be sufficient for neural decoding tasks if we apply data augmentation techniques. But there’s another, even more fundamental challenge: human attention as a confunding factor. Just as “Attention is all you need” in computational models <d-cite key="Vaswani2017"></d-cite>, attention is also critical in humans. Our attention is not static: it fluctuates with arousal, fatigue, mood, and motivation <d-cite key="Shen2024, Smallwood2009"></d-cite>, and these states alter how we engage with language and modulate behaviors and neural activity <d-cite key="Smallwood2011, Unsworth2013"></d-cite>. One particularly common and impactful attention state is **mind-wandering (MW)**, when our focus drifts from the task at hand to unrelated thoughts. **People spend 30% to 60% of their daily lives mind-wandering** <d-cite key="Killingsworth2010"></d-cite>, and it happens frequently during reading <d-cite key="Smallwood2011"></d-cite>. You might even find yourself mind-wandering while reading this blog post. While the exact cognitive processes behind MW are still unclear, one leading hypothesis, the perceptual decoupling hypothesis <d-cite key="Smallwood2006"></d-cite>, suggests that internal thoughts during MW divert resources away from external stimuli. This diversion can directly affect how language is processed. For these reasons, accounting for MW is not just interesting; it is essential for building models that can mechanistically and predictively capture real human language processing.  
### 1.4 Mind-wandering is sneaky, so how do we study it?
#### 1.4.1 Previous approaches and their drawbacks
Now let’s take a brief detour into psychology. Studying constructs like MW can include unique challenges. Prior research has primarily relied on self-reports and thought probes to detect episodes of MW. Self-reports depend on participants’ subjective awareness: whenever individuals realize that their minds have drifted from the task, they are instructed to report it <d-cite key="Schooler2002"></d-cite>. Thought probes, by contrast, are randomly timed prompts that require participants to indicate whether they are currently on-task or mind-wandering <d-cite key="Giambra1995, Smallwood2006"></d-cite>. While both approaches are widely used in contexts such as reading <d-cite key="Reichle2010, Broadway2015, Faber2017"></d-cite>, they only mark when MW ends, making it difficult to capture its onset or to characterize how episodes naturally unfold over time, temporal features that are crucial for understanding and detecting MW. 
#### 1.4.2 Our novel ReMind paradigm
So how did we solve this problem? I spent days reading research articles, and my own wandering mind made me reread passages over and over just to comprehend the material. Suddenly, it hit me: **when we reread after getting lost, we are implicitly marking the parts of the text where our attention drifted.** That observation inspired the ReMind paradigm, which estimates the onset and duration of MW episodes during reading by combining retrospective self-reports with eye-tracking. Participants indicate the words where they believe their mind started and stopped wandering for each episode. We then align these selections with gaze timestamps to estimate precise onset and offset times. 


### 1.5 Introducing the ROAMM dataset
Using this approach, we created the Reading Observed at Mindless Moments (ROAMM) dataset, which contains simultaneous EEG and eye-tracking data from 44 participants (30 females, 9 males, and 5 non-binary) performing naturalistic reading in the ReMind paradigm. The dataset includes rich labels, such as the word at each fixation, attention state at each sample, and a comprehension question score for each page. By capturing attention states in a naturalistic and uninterrupted way, ROAMM provides a powerful resource for studying questions in language and cognition, and for developing language-based machine learning models that better reflect real human attention. 

In the remainder of the post, we present the ROAMM Dataset including the participant demographics, task, data acquisition and pre-processing, and methodology for identifying mind wandering. We include figures of data validation that demonstrate the quality and temporal dynamics of the data. Finally, we describe the availability of the data and discuss potential applications including open questions for ML practitioners and examples of candidate models. 

## 2. ROAMM dataset
### 2.1 Participants
We recruited 58 participants from the University of Vermont who were fluent in English and reported no family history of neurological disorders or epilepsy. All participants underwent screening and provided informed consent before participation. The study protocol was approved by the university Institutional Review Board. 14 participants were excluded due to issues such as equipment difficulties, incomplete experimental runs, monocular-only eye-tracking data, or missing demographic information. The final dataset includes **44 participants**. The participants' age ranged from 18 to 64 years (Mean = 22.6, SD = 7.8, Median = 20, Mode = 19). This indicates a relatively young but moderately varied sample. Our sample also includes labels for gender, handedness, and self-identified ADHD.  

<img src="{{ '/assets/img/2025-11-24-reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/demographics.png' | relative_url }}" alt="demographics" style="zoom:45%;" />

### 2.2 The ReMind paradigm
Participants read five articles selected from Wikipedia (2015): **Pluto** (the dwarf planet), **the Prisoner’s Dilemma**, **Serena Williams**, **the History of Film**, and **the Voynich Manuscript**. These topics were chosen to be unfamiliar yet comprehensible without prior background knowledge. Each article was standardized by removing images and jargon, then divided into 10 pages (≈220 words per page). Pages were rendered using a custom Python script into 16 lines of black Courier-font text on a gray background. To encourage engagement and assess comprehension, we created one multiple-choice question per page. Each question was designed to require attention to that page alone.  

![remind_task](remind_task.png)

The reading task was programmed using PsychoPy <d-cite key="Peirce2019"></d-cite>, a platform for developing psychological experiments. Each experimental session consisted of five runs, one for each article. Articles were presented in a randomized order. Before the first run, participants received task instructions and an explicit definition of mind-wandering (*see above figure*). Participants read at their own pace with no time limit per page but could not re-read previous pages. If they noticed themselves mind-wandering, they pressed the “F” key to access a dedicated reporting screen. There, they clicked on the words marking where they believed the MW episode began and ended (highlighted onscreen for clarity). If the episode began on the prior page, they marked the first word of the current page. After submitting the report, they returned to the same page to resume reading. For consistency, only one MW report was permitted per page. 

### 2.3 Data acquisition and preprocessing 
#### 2.3.1 Eye-tracking
The experiment was conducted in a soundproof booth to minimize distractions. We used an **SR Research EyeLink 1000 Plus eye tracker** to record binocular eye movements and pupil area at 1000 Hz. Each run began with calibration, repeated until EyeLink reported good accuracy (worst error < 1.5°, average error < 1.0°). Once calibration was complete, we recorded eye movements in sync with EEG while participants performed the reading task. PsychoPy sent page-onset triggers to both systems to keep the data streams aligned.  

EyeLink automatically detected fixations, saccades, and blinks using default thresholds. These events were parsed into data frames, and fixations were mapped to individual words on the screen using spatial coordinates. Because pupil size data can be unreliable around blinks (due to eyelid occlusion), we corrected for this by linearly interpolating pupil size using values from the saccades surrounding each blink.

#### 2.3.2 EEG
We recorded simultaneous EEG using a **BioSemi ActiveTwo 64-channel system** at 2048 Hz. Before starting the task, we ensured all electrodes had stable connections by checking impedances and correcting any channels with unusually high values. Collected data were preprocessed in EEGLAB <d-cite key="Delorme2004"></d-cite>: resampled to 256 Hz, re-referenced, filtered (0.5–50 Hz), and channels with poor signal quality interpolated. Eye and muscle artifacts were removed using independent component analysis (ICA) with standard EEGLAB parameters.

### 2.4 Dataset format
We made the ROAMM dataset easy to work with by aligning eye-tracking data to 64-channel EEG at 256 Hz. We downsampled the eye-tracking data using the real-time arrays: for each EEG time point, we identified the closest corresponding eye-tracking sample and used the pupil size at that moment. Fixations, saccades, and blinks were directly mapped using their start and end times relative to the EEG time array.

All data are stored in pandas DataFrames (.pkl format, [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)) for fast loading and smaller file size (compared to .csv format). Each participant has one .pkl file per run, with a total of 5 runs. Eye-tracking events like fixations, saccades, and blinks are expanded across their start-to-end times. For example, if a fixation occurs from 10 to 11 seconds, all samples within that 1-second window are annotated with `is_fixation = 1`, `fix_tStart = 10`, `fix_tEnd = 11`, `fix_duration = 1`, etc. Time stamps, page boundaries, and mind-wandering episodes are all included, along with metadata such as sampling frequency, run numbers, page numbers, and story names. 

To maintain clarity and focus on natural reading, we defined **first-pass reading** as the period when participants were initially reading the text. Activities such as reading task instructions, marking mind-wandering pages, rereading after a mind-wandering report, or answering comprehension questions were excluded from this category. Each sample is labeled for first-pass reading, mind-wandering, and fixated words. Each fixated word also includes a key linking it to the original text, making it easy to generate embeddings or other vectorized representations within the context of the reading corpus for computational modeling. Additional information, including **subject demographic information** and **comprehension question scores** (with corresponding  run and page numbers), and **EEG channel locations** is saved in separate files for easy access.

| Data Category    | Column Count | Description                                                 | Example Columns                                              |
| ---------------- | ------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| **EEG**          | 64           | 64 electrode EEG signals                                    | `Fp1, AF7, AF3, F1, F3...`                                   |
| **Eye-Tracking** | 38           | Gaze position, pupil size, fixations, saccades, and blinks  | `is_fixation, fix_eye, fix_tStart, fix_tEnd, fix_duration...` |
| **Time**         | 8            | Timestamps, page boundaries, durations, and MW episode info | `time, page_start, page_end, page_dur, mw_onset...`          |
| **Others**       | 4            | Sampling frequency, run and page numbers, story name        | `sfreq, page_num, run_num, story_name`                       |
| **Labels**       | 4            | First-pass reading, MW, and fixated word annotations        | `first_pass_reading, is_mw, fix_fixed_word, fix_fixed_word_key` |


### 2.5 Dataset scale
The ROAMM dataset is large and rich. It contains over 46 million recorded samples from 44 participants, totaling more than 50 hours of simultaneous EEG and eye-tracking data. Of these, over 26 million samples (around 30 hours) correspond to first-pass reading, which includes fixated word information at each sample. Across the 2,200 pages read, participants reported 998 mind-wandering episodes, corresponding to 45.4% of the pages. These episodes had a median duration of 5.92 seconds and a mean of 7.79 seconds and resulted in a total of 2.2 hours of reading time annotated as mind-wandering.  

| Data Type         | Total Sample Count | Total Time (h) | Subject Avg Time (m) |
|-------------------|--------------------|------------|-------------|
| **Total Recording** | 46,371,584         | 50.3     | 68.6      |
| **First-Pass Reading** | 26,691,014     | 29.0     | 39.5      |
| **Mind-Wandering** | 2,045,021          | 2.2      | 3.0       |
| **Fixation**       | 38,324,700         | 41.6     | 56.7      |
| **Saccade**        | 9,326,633          | 10.1     | 13.8      |
| **Blink**          | 2,290,418          | 2.5      | 3.4       |

The histograms below illustrate the distribution of data across participants for each attribute in the ROAMM dataset. While all participants contributed, individual differences are evident in the distributions. This highlights the real-world variability in human data and underscores the importance of carefully considering modeling approaches, whether developing a general model across participants or an individualized classifier tailored to each person. 

<img src="{{ '/assets/img/2025-11-24-reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/data_scale.png' | relative_url }}" alt="data_scale" style="zoom:20%;" />

### 2.6 Data validation
We validated the ROAMM dataset to ensure high recording quality, precise alignment between EEG and eye-tracking data streams, accurate fixation-to-word mappings, and reliable labeling of MW episodes. 
#### 2.6.1 EEG and eye-tracking recoding quality
To assess recording quality, we inspected EEG and eye-tracking signals in parallel. As demonstrated below using a randomly selected 10-second window, preprocessed EEG from selected channels show **clean activity with minimal muscle and eye artifacts**. Eye-tracking features behave as expected: **fixations are followed by saccades, blinks appeared distinctly**, and **gaze position traces reveal typical reading patterns**. Specifically, x-coordinates increase left to right across each line, while y-coordinates step down across successive lines, confirming naturalistic line-by-line reading. For context, the top left of the displayed reading page is (x,y) = (258, 5.4) and the bottom right is (x,y) = (1662, 1074.6). 

![eeg_eye_valid](eeg_eye_valid.png)

#### 2.6.2 EEG and eye-tracking alignment
Next, we validated alignment between EEG and eye-tracking streams. Using the unfold toolbox, we deconvolved fixation-related potentials (FRPs) during periods of reading without MW. **The resulting FRPs and P1 topography replicated patterns reported using the ZuCo 2.0 dataset** <d-cite key="Hollenstein2019"></d-cite>, providing strong evidence for the temporal precision of our co-registered recordings. 

![frp](frp.png)

#### 2.6.3 Fixation-to-word mapping
We also validated fixation-to-word mappings by plotting gaze traces directly on reading pages. In one example, a participant read mindfully without reporting MW; in another, the same participant reported an MW episode. Onset and offset words of the MW episode were highlighted in red, while fixations appeared as colored dots. Larger dots indicated longer durations, and a purple-to-yellow gradient reflected temporal order. Fixations within MW episodes were additionally center-colored in red, and consecutive fixations were linked by red lines to mark saccades. **Most fixations aligned neatly with words, and the gaze traces showed clear left-to-right reading flows, confirming the accuracy of fixation-to-word mapping.** 

![reading_page_full](reading_page_full.png)

#### 2.6.4 Reliable MW onset
Finally, we validated MW onset labeling. In a paper currently under review, we demonstrated that incorporating MW onset information significantly improves the performance of linear regression classifiers trained to detect MW from eye-tracking features. A sliding-window analysis not only replicated prior findings of reduced fixation rates during MW episodes <d-cite key="Reichle2010"></d-cite>, but also revealed that these changes begin precisely at the reported MW onset. These findings demonstrate that the **ReMind paradigm provides a powerful framework for capturing MW onset and its progression over time**, ensuring that our attention state annotations are precise and grounded in reliable MW onset information. 

<img src="{{ '/assets/img/2025-11-24-reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/eye_mwonset.png' | relative_url }}" alt="eye_mwonset" style="zoom:40%;" />

### 2.7 Data accessibility and availability
The processed datasets are publicly available on the [OSF](https://osf.io/kmvgb/overview). Due to their large size, raw datasets are not hosted online but are available upon request. All preprocessing scripts used to generate the processed datasets are available in the [GitHub repository](https://github.com/GlassBrainLab/ROAMM.git). 

### 2.8 How to use ROAMM
We put a lot of effort into making ROAMM easy to use, even if you’ve never worked with EEG or eye-tracking data before. Everything is pre-aligned and stored in **pandas DataFrames**, so you can load it with just a few lines of Python. 

Once you download the dataset from [OSF](https://osf.io/kmvgb/overview), here’s how you can get started: 

**1. Import packages and set up paths**

```python
import os
import pandas as pd

# define data root
# this is the path to the ROAMM folder on local machine
roamm_root = r"/Users/~/ROAMM/" # change this to your path
ml_data_root = os.path.join(roamm_root, 'subject_ml_data')
```
**2. Load a single run for one subject**

```python
subject_id = 's10014'
subject_dir = os.path.join(ml_data_root, subject_id)
run_number = 1
df_sub_single_run = pd.read_pickle(os.path.join(subject_dir, f'{subject_id}_run{run_number}_ml_data.pkl'))
```

**3. Load all runs for one subject**
```python
pkl_files = [f for f in os.listdir(subject_dir) if f.endswith('.pkl')]
df_sub_all_runs = pd.DataFrame()
for pkl_file in pkl_files:
    df_sub_single_run = pd.read_pickle(os.path.join(subject_dir, pkl_file))
    df_sub_all_runs = pd.concat([df_sub_all_runs, df_sub_single_run])
```

**4. Load all subjects, filtered to first-pass reading**

```python
# load all runs for all subjects
all_subjects = [d for d in os.listdir(ml_data_root) if d.startswith('s') and os.path.isdir(os.path.join(ml_data_root, d))]
df = pd.DataFrame()
for subject_id in all_subjects:
    subject_dir = os.path.join(ml_data_root, subject_id)
    pkl_files = [f for f in os.listdir(subject_dir) if f.endswith('.pkl')]

    # make sure each subject has 5 runs of data
    if len(pkl_files) != 5:
        raise ValueError(f"Subject {subject_id} has {len(pkl_files)} runs instead of 5")
    
    for pkl_file in pkl_files:
        df_sub_single_run = pd.read_pickle(os.path.join(subject_dir, pkl_file))
        # I highly recommend you to filter out reading runs that are not the first pass reading
        # to save memory
        df_sub_single_run = df_sub_single_run[df_sub_single_run['first_pass_reading'] == 1]
        # add subject id to the dataframe   
        df_sub_single_run['subject_id'] = subject_id
        # convert bool col explicitly to avoid pandas warning
        for col in ['is_blink', 'is_saccade', 'is_fixation', 'is_mw', 'first_pass_reading']:
            df_sub_single_run[col] = df_sub_single_run[col] == True
        # append to the dataframe
        df = pd.concat([df, df_sub_single_run])
    
    print(f'Subject {subject_id} has been loaded.')
```

## 3. Open questions for ML practitioners using ROAMM

The ROAMM dataset is rich in scope, combining multiple valuable modalities: **eye-tracking** (gaze position and pupil size), **brain signals** (i.e., EEG), **human attention states**, and **linguistic content** (the reading text itself). This multimodal design provides countless opportunities for machine learning practitioners to explore how these signals interact. Below, we highlight 4 open questions that showcase the potential of ROAMM for advancing both cognitive science and computational modeling.

<img src="{{ '/assets/img/2025-11-24-reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/roamm_modalities.png' | relative_url }}" alt="roamm_modalities" style="zoom:40%;" />

### 3.1 Learn shared representation from EEG and eye-tracking
Eyes, particularly pupil size, reveal much about internal cognitive states and arousal <d-cite key="Castellotti2025"></d-cite>. With a dataset at the scale of ROAMM, we can now ask whether eye-tracking and EEG share common patterns of variance that allow them to be tightly linked.

One way to explore this question is to use a multi-modal embedding method like CLIP <d-cite key="Radford2021"></d-cite>, which was originally applied to learn joint representations of images and text via contrastive learning. A **CLIP-style** model could be trained on ROAMM by contrasting matched and mismatched EEG and eye-tracking segments. If two modalities indeed have features that share significant variance, the resulting embeddings would enable accurate cross-modal classification. Beyond classification, shared representation opens the door to conditional decoding from one modality to the other. For example, one can train a **diffusion transformer model (DiT)** <d-cite key="Peebles2022"></d-cite> to reconstruct a subject’s EEG signals using eye-tracking data as a generative prior. Since there exists a shared representation space of EEG and eye-tracking, the DiT can fuse the information from eye-tracking data through cross-attention.

### 3.2 Build a momentary human attention decoder
EEG and eye-tracking are not the only modalities in ROAMM that may exhibit strong associations. Previous studies have trained models to detect attention states from EEG and eye-tracking separately. We hypothesize that, when combined together, they can provide a more reliable estimate of a subject’s attention state during reading.

To evaluate this hypothesis, one could train simple classifiers such as regression models, SVM, gradient boosting, or neural models that takes the shared representations of EEG and eye-tracking at a given moment to predict the subject’s attention state at that same time step. However, attention is not merely a transient experience; it unfolds dynamically over time. Thus, signals from a single moment are unlikely to provide sufficient information for accurate prediction. To capture attention’s temporal dynamics, one can train sequential models like **long short-term memory networks**, **temporal convolutional networks**, or **generative transformers**, which have demonstrated superior performance in modeling text data. These models can be trained to predict future attention states from EEG, eye-tracking, and attention states from both current and prior windows, enabling moment-to-moment attention decoding.

### 3.3 Is human attention what we need for neural decoding?
Decoding neural signals has become a popular topic, with several studies attempting to use EEG to decode text <d-cite key="Liu2024, Wang2024"></d-cite>. However, their main focus was on the attention mechanisms in transformers, but they largely neglected fluctuations in human attention during the task. This can be problematic: during mind-wandering, readers still maintain the outward behavior of reading, moving their eyes from left to right and line by line, but their visual input is disrupted by internal thoughts. Cognitive resources that should be allocated to word recognition and comprehension are instead consumed by spontaneous mental activity. In other words, the brain itself cannot fully follow what the eyes are reading during mind-wandering. This raises a fascinating question: **does knowledge of human attention states improve neural decoding performance?**

A straightforward way to address this question is to train an EEG2text decoder separately on data segments from normal reading versus mind-wandering. **Performance differences between these conditions would reveal whether filtering out periods of “mindless reading” provides a decoding advantage.** If we observe better performance when training only on attentive reading, the result would be intuitive: how could a model recover information that the brain itself fails to process? 

However, the more intriguing possibility is if decoding performance remains similar regardless of attention state. This would suggest that information about the text is encoded at lower levels of the visual or sensory hierarchy. In such a scenario, the model may be able to extract signals from early visual or pre-attentive neural activity that are not available to conscious awareness. This opens up provocative implications: **machine learning models could potentially reveal implicit or subliminal processing of linguistic information in the brain.** In other words, this is not a mind-reader but something at another level: a **subconscious-reader** that can uncover information from your brain even when you are not aware. It’s as if your neurons are whispering secrets that only the model can hear.

Thus, testing whether attention modulates neural decoding performance not only has practical implications for building better brain–computer interfaces, but also addresses fundamental cognitive neuroscience questions about the boundary between unconscious encoding and conscious comprehension.

### 3.4 Use EEG, eye-tracking, human attention, and reading text to predict comprehension
The previous questions focused on the link between pairs of modalities in our dataset. But as the saying goes, *“only children make choices, adults want it all!”* For machine learning experts who are not satisfied with pairwise associations and eager to showcase the full power of multimodal modeling, the next challenge is to use all available modalities from ROAMM: EEG, eye-tracking, attention states, and reading text itself. The task we propose is to predict reading comprehension, using ROAMM’s page-level comprehension scores. While page-level labels are coarse and may not perfectly reflect moment-to-moment understanding, they still provide a valuable proxy for comprehension that can anchor multimodal learning.

This problem requires complex model architectures and training frameworks that can integrate heterogeneous data streams. One can opt for the **traditional fusion methods like early fusion** in which features from all modalities are concatenated and processed jointly within a single parameterized model. Although considered a traditional technique, early fusion remains prevalent in recent large-scale multimodal systems for text and images (e.g., <d-cite key="ChameleonTeam2024, Lin2024"></d-cite> early fusion via concatenation, late fusion via ensembling, or hybrid fusion). Besides fusion, we can also train individual models to embed each modality. Examples of this **late fusion include CLIP and Imagine Bind** <d-cite key="Radford2021, Girdhar2023"></d-cite> which trains transformer encoders to map multi-modal data across into an embedding space. Downstream tasks, such as comprehension prediction, can be done by training lightweight classification on top of the shared embeddings. When applying late fusion on ROAMM, one can follow the existing practice, using the same architecture to embed data from all modalities. Alternatively, they can use a specific architecture with inductive bias that accommodates the invariance present in the data (e.g., graph neural network (GNN) for EEG data <d-cite key="Klepl2023"></d-cite>). 

A model that successfully predicts comprehension from this rich multimodal space would not only advance cognitive modeling, but also contribute to the emerging field of Foundation Models for the Brain and Body (*Yes, another workshop hosted this year*). By integrating physiological, behavioral, and linguistic signals into a single predictive framework, we move closer to general-purpose models of human cognition. A recent Nature study <d-cite key="Binz2025"></d-cite> shows that large-scale multimodal learning can capture human behavior across a wide range of domains. Extending these ideas to ROAMM provides an opportunity to build neurocognitive foundation models during naturalistic reading environments.




## 4. Conclusions

In this work, we introduced the **Reading Observed At Mindless Moments (ROAMM)** dataset, a large-scale, multimodal resource capturing simultaneous EEG and eye-tracking data during naturalistic reading. By using the ReMind paradigm, ROAMM stands out among existing reading datasets by providing a highly naturalistic reading environment, temporally resolved attention labels, and precise alignment between neural and behavioral signals.

We provided an overview of the dataset’s acquisition, preprocessing, and structure, highlighting its **scale, richness, and quality**. Validation analyses confirmed **high-fidelity recordings**, **accurate fixation-to-word mappings**, and **reliable labeling of mind-wandering episodes**, making ROAMM suitable for rigorous cognitive and computational modeling.

Beyond describing the dataset, we outlined a set of open questions that illustrate its potential for advancing both neuroscience and machine learning. These include 1) learning shared representations between EEG and eye-tracking, 2) building moment-to-moment attention decoders, 3) investigating the role of attention in neural decoding, and 4) predicting reading comprehension using fully multimodal data. ROAMM thus provides a unique opportunity to explore the interactions between brain signals, eye movements, attention, and language, enabling development of models that better reflect real human cognition.

In summary, ROAMM not only offers a rich resource for fundamental research on attention and reading but also serves as a platform for developing advanced multimodal machine learning models. By bridging cognitive science and computational modeling, it paves the way toward neurocognitive foundation models capable of capturing complex and naturalistic human behavior.
