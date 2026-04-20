import os

import wespeaker
import numpy as np
import torch
import tqdm
import torchaudio

from dotenv import load_dotenv

from lib.models.module import DiarizationModule


load_dotenv()

# Compatibility patch for some torchaudio builds used with SpeechBrain.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]


class WeSpeakerInferenceModule(DiarizationModule):
    def __init__(
        self,
        wav_segments: list,
    ):
        super().__init__(tag="WeSpeaker Inference")
        self.wav_segments = wav_segments
        self.classifier = wespeaker.load_model("english")

    def _extract_embedding(self, speaker_audio):
        if speaker_audio is None or len(speaker_audio) == 0:
            return None

        audio_tensor = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)
        embedding = self.classifier.extract_embedding(audio_tensor)

        if torch.is_tensor(embedding):
            embedding = embedding.squeeze().detach().cpu().numpy()
        else:
            embedding = np.asarray(embedding).squeeze()

        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)

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