---
layout: post
title: "HMS Competition"
date: 2024-05-09 01:43:18 +0530
categories: Deep Learning,Kaggle
---

HMS DATA:

Overall, 124 raters scored 50,697 EEG segments from 2,711 distinct patients’ EEGs. Among the 124 raters, They
identified a subset of 20 physician experts with subspecialty training who each individually annotated ≥1000 EEG
segments. Among EEG segments annotated by these 20 experts, 9,129 segments received labels from at least 10
independent experts. Based on prior work suggesting that ~10 experts are required to achieve a stable group
consensus1, They designated these segments as having “high quality” labels, meaning that these samples received
enough expert labels so that the consensus label (IIIC pattern type with the most labels) and degree of agreement
amongst experts about the correct label can be assigned with high confidence. The remaining labeled data was
considered “lower quality” e.g., because these segments either had fewer labels or not all raters had fellowship
training. They therefore divided the 50,697 labeled EEG segments into a group of 9,129 segments with “high
quality” labels scored by 20 “top experts”; the remaining set of 30,628 segments had lower-quality labels
collectively received from 119 of the 124 raters. 

They expanded the high- and low-quality sets of EEG segments by adding additional segments belonging to the 
same stationary period of the EEG. This yielded 71,982 segments with high-quality labels from 1,522 patients,
and 111,095 segments with lower-quality labels from 1,950 patients. Nevertheless, in the final division of data
into training and test datasets, any given patient’s data is assigned exclusively to either the training or to the test
dataset. 

The justification for expanding labeled data is as follows:
Empirically, the EEG can be divided into a series of “stationary periods” (SP), within which the pattern of EEG activity is unchanging. These SP can be identified by detecting changepoints within the EEG power, as illustrated in Figure . 
The upper four subplots of Figure show spectrograms from four brain regions (LL = left lateral, RL = right lateral, LP = left para-central, and RP =right para-central). A seizure occurs near the center of the image. Below the spectrogram is shown the sum of
the total spectral power across all four regions, and the SP between changepoints (CPs, identified by a CP  detection algorithm). Raw EEG from three different segments within the SP (region within the central part of the
seizure, between the two central pink vertical lines) are shown below, indicated by their starting times (t1, tC, t2),
where “tC” represents the time at the center of SP shown. Each 10-sec EEG segment within the same SP
demonstrates clear seizure activity, like the central segment. This example illustrates the rationale for assigning
the same label to all EEG segments that fall within the same SP. In our approach, experts label the central EEG
segment, and the same label is then automatically assigned to the remaining segments within the SP, increasing
the number of labeled samples available for model training and evaluation.

![EEG](/assets/HMS1.jpg)



