#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#show: ieee.with(
  title: [Development and evaluation of a modular speaker diarization pipeline],
  abstract: [
    This paper presents the development and evaluation of a modular speaker diarization pipeline. By evaluating different voice activity detection parameters, embedding inference models (Pyannote, SpeechBrain and WavLM), and clustering algorithms (Agglomerative, DBSCAN, GMM), the impact of various architectural choices on the Diarization Error Rate (DER) is analyzed. Through the implementation of a sliding window approach, segment merging, and targeted hyperparameter tuning, the pipeline's DER was improved to 0.196. The results highlight the effectiveness of TDNN-based embeddings combined with Cosine-based Agglomerative Clustering while also demonstrating the limitations of density-based clustering and transformer architectures on short audio segments.
  ],
  authors: (
    (
      name: "Reinhold Brant",
      department: [],
      organization: [DHBW Stuttgart],
      location: [Stuttgart, Germany],
      email: ""
    ),
  ),
  index-terms: ("Speaker diarization", "Pipeline"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
Speaker diarization describes the task of deciding who is speaking at a given moment of time. Speaker diarization systems can be used to analyse speech during meetings or telephone calls. To achieve that, there are multiple steps involved to process an audio input and calculate data on the number of speakers and who is speaking when. The aim of this work is to implement a modular diarization pipeline. Furthermore, we want to evaluate different hyperparameter settings and models to achieve a DER (diarization error rate) of below 0.2, which will be seen as achievable for building a modular pipeline from pretrained models as well as tuning hyperparameters. State-of-the-art pipelines achieve DER rates under 0.1 @speechbrain @ecapa.
== Setup
As a first step, the project is set up with `uv`, which is a modern Python package manager to handle package installation and Python versioning. To test out different pipelines efficiently, we will implement Python base classes for modules, which will hold different functionalities of the pipeline and also implement pipeline classes, that will be used simplify execution of different pipelines sequentially. This will lead to a more efficient data collection process.
== Dataset
To evaluate the implemented pipelines a sufficient dataset is needed. In this work, the AMI Corpus dataset will be used. This dataset is "a multi-modal data set consisting of 100 hours of meeting recordings" @ami_corpus. It consists of different meetings with recordings for every speaker per meeting and an audio mix containing the mix of all speakers of the meeting, which will be used in this work. In addition to that, the dataset provides reference data as XML files, which will be used to calculate the DER against our pipeline hypotheses.
== Metrics
To evaluate implemented pipelines, the DER (`diarization error rate`) will be used, as it is the standard evaluation metric of diarization systems @pyannote_metrics_diarization. The DER is calculated with the following formula and a forgiveness collar $c = 0.25s$:
$
  "DER" = frac("false alarm" + "missed detection" + "confusion", "total")
$
False alarms are cases, where the system decides that any speaker is currently speaking when in fact there is silence in the audio. Missed detections are cases, where the system decides that there is silence when in fact there is speech. Confusion is the wrong classification of a speaker. The meaning of the DER calculation segments is visualized in @fig:der.
#figure(
  placement: none,
  image("assets/der.png"),
  caption: [DER calculation components @dercalc]
) <fig:der>
= Methods <sec:methods>
== Pipeline development
In order to achieve the aim of speaker diarization with a DER of below 0.2, we will choose an experimental approach. First, we will setup a fully functional pipeline, which will be gradually improved by means of testing alternative models and approaches for different steps of the pipeline. After each change the pipeline will be evaluated by means of calculating the DER. This process will be repeated in iterations until a sufficient DER is achieved for the pipeline. After that, key components of the pipeline will be changed to alternative models or algorithms to compare to the main pipeline. In the end, the implemented pipeline will be compared to other state-of-the-art methods in terms of DER.
== Pipeline result visualization
To better understand the clustering results and validate the quality of speaker embeddings, visualization techniques are employed. The pipeline uses dimensionality reduction methods such as *Principal Component Analysis (PCA)* to project high-dimensional embedding vectors (typically 512-dimensional) into a 2D or 3D space suitable for visualization. PCA is computationally efficient and preserves global structure. The reduced embeddings are then plotted with color-coding by cluster assignment, allowing for visual inspection of cluster separation and identification of potential outliers or misclassified embeddings. This visualization helps identify parameter tuning issues early in the development process and provides qualitative validation of the clustering performance before evaluating quantitative metrics like DER. In addition to that, the visualization includes the feature, to click on data points and listen to their respective audio chunk from the dataset for acoustic validation.

== Pipeline evaluation
All pipelines are evaluated by DER with the use of the Python package *simpleder* @simpleder.

= Initial Pipeline implementation
== Basic architecture
To first set up a working diarization pipeline we will focus on a classic modular pipeline approach. This modular approach will consist of voice activity detection (VAD), feature extraction, clustering and post processing (see @pipe). The pipeline will take an audio path to the AMI mixed audio sample as input, run the pipeline on the audio and output a label array with annotations on which speaker spoke when. This array can be converted to RTTM (Rich Transcription Time Marked) format.

The following diagram shows the modular approach with which the pipeline will be implemented.
#let pipeline_box(pos, label, tint: white) = fletcher.node(
	pos,
	align(center, label),
	width: 30mm,
	height: 9mm,
	fill: tint.lighten(60%),
	stroke: 1.5pt + tint.darken(20%),
	corner-radius: 5pt,
)

#let pipe = fletcher.diagram(
	spacing: 10pt,
	cell-size: (5mm, 5mm),
	edge-stroke: 1.5pt,
	edge-corner-radius: 5pt,
	mark-scale: 30%,
  label-size: 0.8em,

	// Pipeline components from top to bottom
	pipeline_box((0, 0), [Audio Input], tint: white),
	fletcher.edge("->"),
	
	pipeline_box((0, 1), [VAD], tint: white),
	fletcher.edge("->"),

	
	pipeline_box((0, 2), [Representation Extraction], tint: white),
	fletcher.edge("->"),
	
	pipeline_box((0, 3), [Clustering], tint: white),
	fletcher.edge("->"),
	
	pipeline_box((0, 4), [Label Output], tint: white),
)
#figure(
  pipe,
  caption: "Modular diarization pipeline"
) <pipe>

== Voice Activity Detection
In order to detect who is speaking when, the system has to detect whether there is speech or silence. For that, a voice detection system can be used. There is a variety of VAD systems that are used in industry solutions. To determine a first method to setup a VAD module in the pipeline, we can compare different VAD solutions with help of a precision-recall diagram. The precision on the y-axis is determined by what percentage the model is correct when classifying an audio signal as speech. On the x-axis the recall is determined by how many of the actual positives were found by the model. Any point on a line in the diagram represents a threshold. By lowering the threshold the model will find nearly all positives but have a decrease in precision. By raising the threshold the model will find less false positives but miss some true positives (see @fig:silero). Among the possible solutions presented in the diagram we chose the Silero VAD module to setup a first pipeline prototype, as it has a good precision with higher recall values (see @fig:silero).

#figure(
  placement: none,
  image("assets/silero.png", width: 80%),
  caption: [VAD precision-recall diagram @silero]
) <fig:silero>

The Silero model can be installed by using `uv add silero-vad` (analog to pip usage) and loaded into the system with `load_silero_vad()`.
To get the speech timestamp segments the function `get_speech_timestamps()` will be used. This method has a number of hyperparameters that are crucial to tune for good perfomance (see later). The output of the VAD model is an array of dictionaries with the following format @silero:

\

 `[{"start": time_start * SAMPLING_RATE, "end": time_start * SAMPLING_RATE}]`.

\
== Convert segment to audio
Now that the VAD system predicted speech timestamps for a given audio, the system needs to cut the audio chunks from the original audio with respect to the calculated timestamps. This is done by converting the audio to a Numpy array and slicing it by the product of the *timestamp* and the *sampling rate*.
==  Feature extraction
In general, to categorize speakers the system needs to extract features from the speech chunks of the audio. These features will be used to build clusters. In order to build features from speech chunks, we will first use a pretrained Pyannote embedding model as it is quick to set up. 
\ \
This model is an x-vector TDNN-based model to build so called embedding vectors from audio chunks. The TDNN architecture of this model is based on deep neural networks. These are large mathematical models constructed of many "neurons" resembling the human brain @dnn.
In a classical setup extraction systems create a vector of a speakers voice based on the input audio (i-vectors) @tdnn. These i-vectors are built by using Gaussian-Mixture-Models (GMM) with matrix projection @tdnn. These vectors are a generalized representation of speech and can be imprecise. This can be improved by x-vectors, which are output of a DNN. The DNN is trained to discriminate between speakers directly. This way, the vector representations are more suitable for clustering @tdnn. A TDNN is a Time-Delay Neural Network, which also considers time based relationships between data to compute x-vectors. It looks at multiple points in time to build an embedding from a time window instead of a frame @tdnn.
\ \
The Pyannote embedding extractor can be set up by loading a pretrained model from Huggingface @emb-hf-pyannote. It takes a Numpy array built from a wav-audio segment as input and outputs a $1 times 512$ embedding vector. This is done for all audio segments built from the VAD timestamps. We get a $N times 512$ x-vector array which can be used for clustering.

== Clustering <clustering>
Clustering algorithms group similar embedding vectors to assign them to individual speakers. Three common approaches for speaker diarization are:

*DBSCAN* groups embeddings based on local density. It identifies clusters by connecting points that are within distance $epsilon$ of each other and requires at least `min_samples` points per cluster. Points that do not belong to any cluster are marked as noise. DBSCAN does not require specifying the number of clusters beforehand but can struggle with varying cluster densities @DBSCAN.

*Agglomerative Hierarchical Clustering* iteratively merges the closest pairs of embeddings or clusters based on a linkage criterion (e.g., average, complete linkage). It produces a dendrogram that can be cut at different heights to obtain the desired number of clusters. This approach is stable and works well when the speaker count is known or estimated in advance @ahc.

*Gaussian Mixture Models (GMM)* model the embedding space as a mixture of Gaussian distributions, where each Gaussian represents a speaker. The model learns the mean, covariance, and mixing weights of each Gaussian through expectation-maximization (EM). GMM-based clustering provides probabilistic assignments and is particularly effective for speaker diarization, as it naturally captures the statistical properties of speaker embeddings @gmm.
These methods will be used to test which algorithm yields best results in the pipeline.
\ \
To cluster embedding vectors in the initial pipeline, DBSCAN will be used.
\ \
#figure(
  placement: none,
  image("assets/init-visu.png", width: 80%),
  caption: [VAD precision-recall diagram @silero]
) <fig:init-visu>
It can be observed that the majority of segments are classified as noise. Furthermore upon further analysis, which is a qualitative analysis listening to the segments and their label, it becomes apparent, that this pipeline can be further optimized. The DER of this pipeline is dependent on the embedding and VAD variance but can be pinned down to $"DER" approx 0.55$, which is not sufficient.

= Pipeline improvement
In the initial implementation of the modular diarization pipeline it became apparent, that the quality of detections is not sufficient. In order to improve the pipeline, it is reasonable to start with the key data the rest of the modules depend on, which is the VAD segments.
== Tuning VAD parameters
The initial parameters for the VAD system were as follows:
- *Threshold*: the higher the threshold the stricter the VAD system is with respect to what is detected as speech and what is detected as silence. This was set at *0.5*.
- *min_speech_duration*: Determines the minimum length of a speech segment. This was set at *400ms*.
- *max_speech_duration*: Determines the maximum length of the speech segment. This was set to *1.5s*.
- *min_silence_duration*: Determines the minimum amount of silence between speech segments. This was set to *500ms*.
- *speech_pad_ms*: This adds a padding before and after the segment to account for more context information. This was set to *20ms*.

After evaluating the initial pipeline it seemed reasonable to set *max_speech_duration* to *3 seconds*. This way, the system would not cut off speech chunks in the middle of an utterance. Otherwise, the embedding extraction would not produce consistent results. This improvement was made with the later idea to split detected segments with a sliding window approach to not yield too long segments.
In addition to that, the min speech duration was reduced to *250ms*. This parameter change was tested with *DBSCAN* and *C-space* clustering and yielded following results:

#figure(
  table(
    columns: 3,
    align: (center, center, center),
    [*Method*], [*DER (initial)*], [*DER (improved)*],
    [DBSCAN], [0.55], [0.49],
    [C-Space (Agglomerative)], [0.51], [0.46],
  ),
  caption: "DER comparison across different clustering methods and VAD tuning"
) <tab:der-comparison>
Although this did not result in a large improvement of the DER, we could reduce the DER, which was an improvement. All further improvements were made on the basis of C-Space clustering.
== Merging segments
Merging adjacent VAD segments reduces fragmentation caused by brief pauses within a single speaker's turn. By combining consecutive segments of the same speaker into longer blocks, the embedding extraction process receives more context, leading to more stable and discriminative speaker representations. We now ran a test to evaluate the DER with this change, which resulted in a DER decrease of around *3%*. Again, this change would not be sufficient to improve the pipeline significantly.
== Sliding window
Instead of clustering entire VAD segments (which can be long and contain multiple speakers), the sliding window approach divides all segments into fixed-length overlapping windows (e.g., 1.5 seconds with 0.75-second step). This ensures that each embedding corresponds to a pure single-speaker utterance.

#figure(
  table(
    columns: 2,
    align: (center, center),
    [*Method*], [*DER*],
    [C-Space (Agglomerative)], [0.46], [C-Space with Merging], [0.42], [C-Space with sliding window], [0.43]
  ),
  caption: "DER comparison across different clustering methods with sliding window"
) <tab:sliding>

In a further step, merging and sliding was combined to yield a $"DER" approx 0.36$, which was an improvement.

== Second VAD parameter tuning
With sliding windows in place to ensure consistent, short segment length, the maximum speech duration of the VAD module was set to infinite in order to not have utterances cut off. This improved the DER significantly to a $"DER" approx 0.22$. Further reducing the step size of the sliding window to *0.5s* improved the DER to *0.196*. In general, the pipeline was improved through a combination of architectural changes and tuning of parameters.

#figure(
  placement: none,
  image("assets/merge_slide.png", width: 80%),
  caption: [DER 0.196 pipeline clusters with merging and sliding window]
) <fig:merge-slide>

It can be observed that the clusters look more dense than with the initial pipeline. Moreover, sampling the data points acoustically works better.

#import "@preview/lilaq:0.6.0": *

= Testing different modules
In general, there are two main module types that can be improved in the pipeline. On the one hand there is the embedding vector inference, on the other hand there is the clustering. The Silero VAD module will not be replaced with other modules as it produces robust results.
== Comparing inference models 
In this chapter, the Pyannote x-vector TDNN based inference used in the initial pipeline will be compared with the ECAPA-TDNN based architecture of the SpeechBrain embedding inference @speechbrain-url @speechbrain. Emphasized Channel Attention, Propagation and Aggregation - TDNN (ECAPA-TDNN) is another form of a TDNN based architecture that also emphasizes important channels for feature extraction. The model has learned which channels tend to be important for the feature extraction and applies that knowledge to the features generated by the TDNN. In addition to that, it connects multiple layers to aggregate information @ecapa.
Compared to the Pyannote inference, the Speechbrain ECAPA-TDNN based inference could not perform as well. This is caused by the pipeline architecture, as merging and sliding windows may cause the audio chunks to be too short for embedding extraction. With merging and sliding window the SpeechBrain inference achieved $"DER" approx 0.55$, without merging and sliding $"DER" approx 0.49$. With suitable changes to the pipeline and tuning, the ECAPA-TDNN based architecture can achieve DER metrics of around *0.03* on the AMI corpus @speechbrain. In addition to above models, *Wav2Vec* was tested to extract embedding vectors.
\ \

#align(center)[
  #table(
    columns: 2,
    align: (left, right),
    [*Inference Model*], [*DER*],
    [Pyannote (TDNN)], [0.196],
    [SpeechBrain (ECAPA-TDNN)], [0.560],
    [WavLM-Base-SV (Transformer)], [0.870]
  )
]

== Comparing clustering methods
The optimized pipeline (using merging and sliding window) was evaluated with three different clustering methods: Agglomerative Clustering (using Cosine Similarity), DBSCAN, and Gaussian Mixture Models (GMM). Because DBSCAN and GMMs struggle natively with the high dimensionality of the embedding vectors, a Principal Component Analysis (PCA) was applied to reduce the dimensionality down to 5 components before clustering. 

Despite this dimensionality reduction, the results show a massive performance gap:
\ \

#align(center)[
  #table(
    columns: 2,
    align: (left, right),
    [*Clustering Method*], [*DER*],
    [Agglomerative (Cosine)], [0.196],
    [GMM (with PCA)], [0.815],
    [DBSCAN (with PCA)], [0.873]
  )
]
= Results and discussion
Although the initial modular pipeline did not perform according set standards, through architectural changes and tuning of hyperparameters, the DER of the modular pipeline could be decreased to $"DER" approx 0.196$, which is a satisfactory result for a modular pipeline without sophisticated overlap detection. Although VAD parameter tuning yielded small improvements, it was a combination of the sliding window approach with the tuning of hyperparameter values, that improved the pipeline the most (see @fig:der-improv).
#figure(
  diagram(
    title: [DER of the modular pipeline],
    ylabel: $"DER"$,
    xaxis: (
    ticks: range(0, 7)
    ),
    ylim: (0, 0.6),
    xlim: (0, 6),
    plot((0, 1, 2, 3, 4, 5, 6), (0.51, 0.46, 0.42, 0.43, 0.36, 0.22, 0.196), mark:"o"),
    plot((-2, 9), (0.2, 0.2), mark: "none")
  ),
  caption: "DER improvement of the modular pipeline\n0 - initial pipeline; 1 - VAD tuning; 2 - C-Space clustering; 3 - Merging; 4 - Sliding & merging; 5 - Further VAD tuning; 6 - Tuning sliding window parameters",
) <fig:der-improv>

It has to be said, that the implemented initial pipeline can be further improved by implementing overlap detection and adapting the pipeline architecture to better suit specific inference models as well as adding modules for resegmentation. Still, Pyannote inference scores around a DER of *0.185*, which resembles our results of $"DER" approx 0.196$ on the AMI test set. Also an improvement to $"DER" approx 0.13$ can be achieved with overlap detection @comparison. As mentioned before, DER metrics of around 0.03 can be achieved with SpeechBrain models with an optimal setup @speechbrain. 
\ \
Ultimately, these findings highlight the intrinsic trade-off between pipeline modularity and specialized end-to-end optimization: while modular components offer high flexibility for iterative testing and evaluation, achieving state-of-the-art performance typically requires highly integrated systems.
