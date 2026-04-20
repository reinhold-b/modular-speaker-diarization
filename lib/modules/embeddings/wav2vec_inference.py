import os

import numpy as np
import torch
import tqdm
import torchaudio

from dotenv import load_dotenv
from speechbrain.inference.classifiers import EncoderClassifier

from lib.models.module import DiarizationModule
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector


load_dotenv()

# monkey patch for some torchaudio builds used with SpeechBrain
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]


class Wav2VecInferenceModule(DiarizationModule):
    def __init__(
        self,
        wav_segments: list,
        model_name: str = "microsoft/wavlm-base-plus-sv",
        savedir: str = "pretrained_models/speechbrain_ecapa",
    ):
        super().__init__(tag="Wav2Vec Inference")
        self.wav_segments = wav_segments
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioXVector.from_pretrained(model_name).to("cpu")
        self.model.eval()

    def _extract_embedding(self, speaker_audio):
        if speaker_audio is None or len(speaker_audio) == 0:
            return None

        waveform = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.embeddings

        if torch.is_tensor(embedding):
            embedding = embedding.squeeze().detach().cpu().numpy()
        else:
            embedding = np.asarray(embedding).squeeze()

        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)

        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

    def run(self):
        embeddings = []
        for segment in tqdm.tqdm(self.wav_segments):
            embedding = self._extract_embedding(segment)
            if embedding is None:
                continue

            embeddings.append(embedding)
            with open("data.csv", "a+") as f:
                np.savetxt(f, embedding.reshape(1, -1), delimiter=",")

        if len(embeddings) == 0:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(embeddings)