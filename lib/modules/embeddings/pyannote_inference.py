import os

import numpy as np
from pyannote.audio import Inference, Model
import torch

from lib.models.module import DiarizationModule
from lib.const import SAMPLING_RATE

import tqdm

from dotenv import load_dotenv
load_dotenv()

class PyannoteInferenceModule(DiarizationModule):
    def __init__(self, wav_segments: list, model_name: str = "pyannote/embedding"):
        super().__init__(tag="Pyannote Inference")

        model = Model.from_pretrained(model_name,
            use_auth_token=os.environ.get("HF_TOKEN")
        )

        self.inference = Inference(model)
        self.wav_segments = wav_segments

    def _extract_embedding(self, speaker_audio):
        if speaker_audio is None or len(speaker_audio) == 0:
            return None

        audio_tensor = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)
        embedding = self.inference({
            "waveform": audio_tensor,
            "sample_rate": SAMPLING_RATE,
        })

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

        