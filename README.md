# Implementation of a Speaker Diarization System

## Overview

This project implements a modular speaker diarization pipeline to determine who is speaking at any given moment in time. The pipeline has been iteratively optimized and achieves a DER (Diarization Error Rate) of approximately 0.196 on the AMI Corpus.

## Installation and Execution

### Requirement: `uv` Package Manager

This project uses **`uv`** as the Python package manager instead of `pip`. This was chosen because `uv` offers the following advantages:

- **Faster Installation**: Parallel downloading and installation of packages
- **Better Dependency Management**: Exact Python version control (Python 3.12)
- **Consistency**: Via `uv.lock`, the exact environment can be reproduced across machines
- **Simple Syntax**: Similar to `pip`, but with modern optimizations

`uv` can be installed at https://docs.astral.sh/uv/getting-started/.

### Access tokens
Please visit https://huggingface.co/settings/tokens to create an access token.
Paste it into a `.env` file in the projects base directory (where `pipeline.py` is located.) as:
```
HF_TOKEN=....
```
To run some models, you may need to accept their terms and conditions, but you will be prompted in the terminal if that is the case.

### Running the Pipeline

To execute the diarization pipeline:

```bash
uv run pipeline.py --audio_path <path_to_audio> [--visualize]
```

The pipeline performs the following steps:
1. Voice Activity Detection (VAD) using Silero VAD
2. Audio segmentation and windowing
3. Speaker embedding extraction (Pyannote TDNN)
4. Clustering of embeddings (Agglomerative Clustering with Cosine Similarity)
5. Post-processing and label output

## Dataset: AMI Corpus

### Download and Structure

The **AMI Corpus** is a publicly available dataset containing over 100 hours of meeting recordings. The files are structured as follows:

```
datasets/
└── amicorpus/
    ├── ami_public_manual_1.6.2/
    │   └── segments/               # XML reference data for DER calculation
    ├── ES2015a.Mix-Headset.wav     # Test audio (mix of all speakers)
    └── ES2016a.Mix-Headset.wav     # Test audio (mix of all speakers)
```

### Download

The dataset can be downloaded from https://groups.inf.ed.ac.uk/ami/corpus. The audio files should be placed in the `datasets/amicorpus/` directory. The XML reference data is located in the `ami_public_manual_1.6.2/segments/` folder and is used for DER calculation.

### Audio Files

Two meeting audio files are used for evaluation:
- **ES2015a.Mix-Headset.wav**: Mix recording of meeting ES2015a
- **ES2016a.Mix-Headset.wav**: Mix recording of meeting ES2016a

These are 16 kHz mono audio files representing the mix of all speakers from each meeting.

## Absolute Paths

**⚠️ Warning: The project contains hardcoded absolute paths**

The following files contain absolute paths that need to be adjusted to your local machine:

- **`lib/utils/xml_to_ref.py`**: 
  ```python
  file_path = base_dir.joinpath("datasets/amicorpus/ami_public_manual_1.6.2/segments")
  ```
These paths must be adjusted when running the project on a different machine. A future improvement would be to use environment variables or relative paths.

## Live System Prototype: `main.py`

The **`main.py`** file implements a live diarization system prototype with real-time audio capture and processing. However, this file currently has **only symbolic value** and is **not fully implemented**.

### Status

- The basic structure for real-time audio capture (via `sounddevice`) exists
- VAD and embedding extraction are present
- The file was intended for interactive testing and experimentation with live audio
- Further development is required for productive use

For actual evaluation and research, **`pipeline.py`** should be used, as this file is fully tested and optimized.

In the **`pipeline.py`** file you will find multiple pipelines that have been used for testing and represent the developemnt of the system. Remove the comments in the `main` method to test multiple pipelines per run.