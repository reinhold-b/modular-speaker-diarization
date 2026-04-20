"""
Pipeline for speaker diarization task.

This module contains the main pipeline for the speaker diarization task, 
including the live system and the embedding extraction process.
"""

import logging
from silero_vad import (load_silero_vad, 
                        VADIterator)
import os
import argparse 
import torchaudio
from lib.const import *


# monkeypatch torchaudio to avoid issues with some builds and SpeechBrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]  

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: None

from lib.log import get_system_logger
from dotenv import load_dotenv
from pyannote.audio import Inference, Model, Pipeline
from lib.utils.xml_to_ref import load_refs_from_audio_file 
load_dotenv()

import simpleder

from lib.const import *

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = get_system_logger()

from lib.modules.vad.silero import SileroVADModule
from lib.modules.embeddings.pyannote_inference import PyannoteInferenceModule
from lib.modules.embeddings.speechbrain_inference import SpeechBrainInferenceModule
from lib.modules.embeddings.wav2vec_inference import Wav2VecInferenceModule 
from lib.modules.helpers.vad_to_wav_segments import VADToWavSegmentsLoader 
from lib.modules.helpers.labels_to_hyp import LabelsToHypModule
from lib.modules.helpers.merge_vad_module import MergeVADSegmentsModule 
from lib.modules.helpers.sliding_window_vad2wav import SlidingWindowVADToWavSegmentsLoader 
from lib.modules.clustering.dbscan_clustering import DBSCANClusteringModule
from lib.modules.clustering.cspace_clustering import CSpaceClusteringModule
from lib.modules.clustering.cspace_improved import CSpaceImprovedClusteringModule 
from lib.modules.clustering.gmm_clustering import GaussianMixtureModelClustering 
from lib.modules.visualization.embedding_visu import EmbeddingVisualizationModule
from lib.modules.overlap.pyannote_overlap_module import PyannoteOverlapModule 

class InitialDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = DBSCANClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der


class CSpaceDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        print(refs)
        print(hyp)

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class CSpaceWithMergeDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        print(refs, hyp)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class CSpaceMergeImprovedDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceImprovedClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments, refs).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        print(refs, hyp)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class SlidingWindowDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments)
        wav_segments, window_timestamps = loader.execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        
        hyp = LabelsToHypModule(labels, window_timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der


class MergingAndSlidingWindowDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        wav_segments, window_timestamps = loader.execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        
        hyp = LabelsToHypModule(labels, window_timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class SpeechbrainMergingAndSlidingWindowDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        #vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        #loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        #wav_segments, window_timestamps = loader.execute()
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class SlidingMergingDBSCANPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        wav_segments, window_timestamps = loader.execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = DBSCANClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, window_timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der
    
class SlidingMergingGaussianMixtureModelPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        wav_segments, window_timestamps = loader.execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = GaussianMixtureModelClustering(embeddings, n_clusters=4).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, window_timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class GaussianMixtureModelDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        wav_segments, window_timestamps = loader.execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = GaussianMixtureModelClustering(embeddings, n_clusters=4).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, window_timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class PrebuildPyannotePipeline:
    """
    Could not get this to work with the latest pyannote version, 
    but it would be interesting to see how good the pretrained pipeline performs on the dataset.
    """
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        waveform, sample_rate = torchaudio.load(audio_path)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=os.getenv("HF_TOKEN"),
        )

        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        with open("pyannote.rttm", "w") as f:
            diarization.write_rttm(f)
        
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class NemoPipeline:
    """
    Could not get this to work with the latest NeMo version.
    """
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")
        from nemo.collections.asr.models import SortformerEncLabelModel
        model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
        model.eval()

        segments = model.diarize([audio_path])
        print(segments)

    def get_der(self):        
        return self.der

class Wav2VecInferencePipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        vad_segments = MergeVADSegmentsModule(vad_segments).execute()
        loader = SlidingWindowVADToWavSegmentsLoader(audio_path, vad_segments, step=0.5)
        wav_segments, timestamps = loader.execute()
        embeddings = Wav2VecInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, timestamps).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.25)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Diarization Pipeline")
    parser.add_argument("--audio_path", type=str, help="Path to the input audio file.")
    parser.add_argument("--visualize", action="store_true", help="Show interactive embedding visualization.")
    args = parser.parse_args()

    p1 = MergingAndSlidingWindowDiarizationPipeline()
    p1.run(args.audio_path, visualize=args.visualize)
    
    #dbscan = SlidingMergingDBSCANPipeline()
    #dbscan.run(args.audio_path, visualize=args.visualize)

    #gmm = SlidingMergingGaussianMixtureModelPipeline()
    #gmm.run(args.audio_path, visualize=args.visualize)

    logger.info(f"DER for Merging + Sliding Window Pipeline: {p1.get_der()}")
    #logger.info(f"DER for DBSCAN Clustering with Merging + Sliding Window: {dbscan.get_der()}")
    #logger.info(f"DER for GMM Clustering with Merging + Sliding Window: {gmm.get_der()}")