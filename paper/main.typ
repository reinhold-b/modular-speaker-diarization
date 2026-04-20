#import "@preview/elsearticle:2.0.2": *

#let abstract = "In todays world, artificial intelligence systems are used in a wide variety of applications. A very important field of use are speech systems, which rely on AI to recognize or generate speech. Another use case is speaker diarization, which are systems designed to recognize different speaker during a conversation and map segments of a dialog to diarized speakers.

The aim of this project is to assemble a modular pipeline for speaker diarization and analyse it's strength using the DER (diarization error rate) and speaker count error as metrics. The analysis will include the evaluation of the pipeline using different modular systemsm for tasks such as Voice Activity Detection (VAD), segmentation, speaker embedding, clustering and post-processing. It is not an aim, to build a speech diarization pipeline from scratch by training models."

#show: elsearticle.with(
  title: "Development and analysis of a modular speaker diarization pipeline",
  authors: (
    (name: [Reinhold Brant], affiliations: (), corresponding: false, email:"author@univa.edu"),
  ),
  affiliations: (
    "": [DHBW Stuttgart],
  ),
  journal: "Name of the Journal",
  abstract: abstract,
  keywords: (),
  format: "review",
  // numcol: 1,
  // line-numbering: true,
)

= Problem

Speaker diarization addresses the question _"who spoke when"_ in an audio recording. While this objective appears simple, robust diarization in realistic settings remains difficult because conversational audio is highly variable. Recordings may contain background noise, reverberation, short speaker turns, non-speech events, and overlapping speech. In many practical cases, the number of active speakers is unknown in advance, which further complicates segmentation and assignment.

The problem targeted in this work is the development of a generic diarization system that can process unseen recordings and assign speech segments to speaker identities without speaker-specific retraining. To make this feasible and extensible, the pipeline should be modular: each stage---voice activity detection, segmentation, embedding extraction, clustering, and post-processing---must be configurable and replaceable.

Beyond achieving low diarization error, an important objective is to understand which parts of the pipeline have the strongest influence on performance and robustness. The system should therefore support systematic comparison of module combinations and parameter settings, using consistent evaluation metrics. In summary, the central problem is to design and analyze a modular speaker diarization pipeline that is accurate, robust across conditions, and flexible enough to adapt to different application domains.



= Theoretical approach
The chosen approach to develop the pipeline will be of experimental nature. First, research regarding usable libraries and systems is conducted. After that, an experimental pipeline will be developed for first tests. The pipeline will be improved and tested with different systems and libraries for each modular step of the pipeline. With each change, the pipeline will be evaluated.

More specific, the following steps will be used as an agenda for the project.

== Dataset research
In a first step, datasets will be researched, that can be used for working with speaker diarization. These should be dialog datasets or podcasts, as they all include two or more people talking and can be used for training models and testing the pipeline.

Several datasets have been found through literature research and will be reviewed @areview:
+ CALL HOME dataset: 500 multilingual online speech meetings with two to seven speakers
+ AMI Corpus: 100 hours of session recordings with three to five speakers
+ CHiME-5/6: 50 hours of real-world conversation
+ VoxSRC: 74 hours of conversation.

== System research
After dataset research, candidate technologies are selected for each module of the diarization pipeline. The goal is to compare classical and neural approaches under the same evaluation setup.

The following system candidates will be considered:

+ *Voice Activity Detection (VAD):* WebRTC VAD for a light-weight baseline @webrtc, Silero VAD for neural frame-level speech detection @silero.
+ *Segmentation / speaker change modeling:* pyannote segmentation models @pyannote2020 and EEND-style overlap-aware segmentation ideas @fujita2019eend.
+ *Speaker embedding extraction:* x-vector embeddings @snyder2018xvectors and ECAPA-TDNN embeddings @desplanques2020ecapa.
+ *Clustering / speaker assignment:* Agglomerative clustering with PLDA-style backends as a standard baseline @sell2018dihard, spectral clustering as an alternative @ng2002spectral, and VBx for Bayesian sequence-constrained clustering @landini2022vbx.
+ *Post-processing and constraints:* resegmentation / smoothing inspired by VB-HMM style diarization backends @landini2022vbx and overlap-aware correction strategies from modern neural diarization literature @fujita2019eend.
+ *Reference implementations for rapid prototyping:* pyannote.audio @pyannote2020, SpeechBrain @speechbrain2021, and NVIDIA NeMo @nemo2021.

These systems provide a broad search space for modular experiments while keeping all module boundaries explicit. This supports controlled ablations, e.g., replacing only the embedding model while keeping VAD and clustering fixed.

== Testing
Now, all systems will be tested with the proposed datasets ultimately evaluating the DER and improving the pipeline.





#bibliography("refs.bib")