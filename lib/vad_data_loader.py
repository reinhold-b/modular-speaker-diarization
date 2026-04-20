import json
import numpy as np
from silero_vad import read_audio

class VADDataLoader:
    """
    This module loads the data created by the VAD pipeline
    into a python object to be used by the embedding system.
    """

    DEFAULT_SAMPLE_RATE = 16_000

    @staticmethod
    def load_wav_as_array(path_audio: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Load wav file from disk and return mono waveform as numpy array."""
        wav = read_audio(path_audio, sampling_rate=sample_rate)

        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        else:
            wav = np.asarray(wav)

        if wav.ndim > 1:
            wav = wav.squeeze()

        return wav.astype(np.float32)

    @staticmethod
    def load_from_json(filepath_vad: str, path_audio: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> list[np.ndarray]:
        """
        Read VAD timestamps from JSON and return list of audio segments.

        JSON format expected from Silero with return_seconds=True:
        [{"start": 1.23, "end": 2.34}, ...]
        """
        with open(filepath_vad, "r") as file:
            data_json = json.load(file)

        if not isinstance(data_json, list):
            raise ValueError("VAD JSON must be a list of timestamp objects.")

        wav = VADDataLoader.load_wav_as_array(path_audio, sample_rate=sample_rate)
        wav_len = len(wav)

        segments: list[np.ndarray] = []
        for segment in data_json:
            if not isinstance(segment, dict):
                continue

            if "start" not in segment or "end" not in segment:
                continue

            start_idx = int(float(segment["start"]) * sample_rate)
            end_idx = int(float(segment["end"]) * sample_rate)

            start_idx = max(0, min(start_idx, wav_len))
            end_idx = max(0, min(end_idx, wav_len))

            if end_idx <= start_idx:
                continue

            segments.append(wav[start_idx:end_idx])

        return segments