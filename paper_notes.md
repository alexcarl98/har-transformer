https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9372813


Human’s move different
- We need an invariant feature learning framework

IFLF incorporates two learning paradigms:
1. Meta-learning to capture robust features across seen domains and adapt to an unseen one with similarity-based data selection.
2. Multi-task learning to deal with data shortage and enhance overall performance via knowledge sharing among different subjects. 


# Intro
Can handle several subjects, several devices.

Hard to re-use data from various datasets

HUMAN activity recognition (HAR) is the foundation to realize remote health services and in-home mobility monitoring. Although deep learning has seen many successes in this field, training deep models often requires a large amount of sensory data that is not always available [1]. 

For research ethics compliance, it takes a long time to design study protocols, recruit volunteers and collect customized sensory datasets. Also, public inertial measurement unit (IMU) sensor datasets on HAR are collected by different researchers following different experiment protocols, making them difficult to be used by others. 

The variability among human subjects and device types in data collection limits the direct reuse of data as well. Deep learning methods have poor generalization ability when testing data (target domain) differs from training data (source domain) due to device and subject heterogeneities (generally known as the _domain shift problem_). 

Fig. 1 shows the effects of cross-domain data variability. 
- Fig. 1a visualizes features from subjects that are seen to the deep model for HAR (left) and as held-out data to the model (right), respectively. 
    - The features are well clustered when subjects are seen to the model and are inseparable for the unseen subject even though she performs the same group of activities wearing the same sensor at the same location.
- Fig. 1b demonstrates the effect of device diversity.
    - The data is collected when a person performs several activities with devices A and B attached to the same on-body locations. 

A deep learning model is trained with device A’s data. We find that despite its high inference accuracy on the testing data from the same device, the accuracy drops drastically on device B’s data. 

In addressing the aforementioned domain shift problems, a pooling task model (PTM) that mixes data from different domains (e.g., subjects, devices) will have low discriminative power as it ignores the dissimilarity among the domains. 

> This is good to know, I was pretty bummed when I tried adding in other data from PAMAP2 and the model performed worse. But this is aligned with what the paper states.

On the other hand, a model trained solely on data from a specific domain requires a lot of training data as it fails to take the advantage of the similarity among different sources. 

Since collecting and labeling sensory data with sufficient diversity is time-consuming, it is impractical to train a task-specific model for each new subject or device encountered from scratch. 

A few previous works have investigated domain shifts caused by device and subject diversity in HAR. In [2] and [3], the problem is formulated as _domain adaptation between a pair of participants or devices_. However, in practice, HAR is rarely limited to transferring knowledge between a pair of domains, rather from a group of source domains (e.g., subjects, placements, or devices) to a target domain. Furthermore, unsupervised domain adaptation approaches trade-off their performance with data labeling efforts. For example, in [2], the authors report an F1-score ≤ 0.8 in testing compared to 0.92 from [4] with supervised learning on the Opportunity dataset [5].


The main contributions of this work are:
- DL framework that handles various sources of domain shifts by extracting domain invariant features across multiple source domains. 
- IFLF alleviates data shortage through feature sharing from multiple source domains. 
- The proposed method achieves superior performance in extensive experiments over multiple datasets than the SOTA meta-learning approach. 
- A similarity metric is proposed to further reduce the amount of labeled data needed from a target domain for model adaptation.


# Related Work

OG Way: 
- ML model trained on hand crafted features
> These are the same features you're feeding into your 'Deep Learning' model


Deep models are reported to achieve state-of-the-art results on many popular open datasets [4], [11], [12], [13]


# Method

Let the input and label spaces be $𝒳$ and $𝒴$, respectively. 
- $𝒳$ input space
- $𝒴$ label space
- $D_{tgt}\{ x_n, y_n ) \}^M _{n=1}$: target domain space
- $D_{src}\{ x_n, y_n ) \}^M _{n=1}$: Label space

The target domain and the set of source domains are $𝒟_{tgt}\{ (x_n, y_n ) \}^M _{n=1}$ and $𝒟_{src}= \{D_1, D_2, ..., D_K\}$, respectively. $𝒟_{tgt}$ and $𝒟_{src}$ follow different distributions on the joint space $𝒳 × 𝒴$.

- A _domain_ $D_{k}=\{ (x_n^{(k)}, y_n^{(k)} ) \}^{N_k} _{n=1}$ corresponds to a source of variation, e.g., a subject or a device, where $N_k$ is the number of labeled data samples.

In HAR, each task is a multi-class classification problem that predicts the activity being performed from data sampled from the respective domain. The problem of meta-learning aims to learn well-generalized features from multiple source domains, and adapts the trained model to the target domain with small amount of labeled data. 

We assume the existence of domain-invariant features across the source and target domains, only the domain specific layers of the model need to be updated when applying to the target domain. In this section, we present the detail of the proposed method. First, we illustrate the overall idea in Section 3.1. Second, the detail of invariant feature learning will be introduced in Section 3.2. Third, we explain in Section 3.3 the similarity-based fast adaptation.


## Overview

intution behind IFLF learn 2 types of knowledge from several source domains:
- learn shared features that can boost the generalization of ml model & 
- task-specific knowledge that provides discriminative power