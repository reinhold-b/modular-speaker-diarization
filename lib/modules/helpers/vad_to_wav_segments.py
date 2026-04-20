from lib.models.module import DiarizationModule
import numpy as np
from silero_vad import read_audio
from lib.const import SAMPLING_RATE 

class VADToWavSegmentsLoader(DiarizationModule):
    def __init__(self, audio_path: str, speech_timestamps: list):
        super().__init__(tag="VAD to WAV Segments")
        self.audio_path = audio_path
        self.speech_timestamps = speech_timestamps

    def load_wav_as_array(self, path_audio: str, sample_rate: int = SAMPLING_RATE) -> np.ndarray:
        """Load wav file from disk and return mono waveform as numpy array."""
        wav = read_audio(path_audio, sampling_rate=sample_rate)

        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        else:
            wav = np.asarray(wav)

        if wav.ndim > 1:
            wav = wav.squeeze()

        return wav.astype(np.float32)

    def run(self):
        """
        Convert VAD timestamps to WAV segments for embedding extraction.
        """
        wav = self.load_wav_as_array(self.audio_path, sample_rate=SAMPLING_RATE)
        wav_len = len(wav)

        segments: list[np.ndarray] = []
        for segment in self.speech_timestamps:
            if not isinstance(segment, dict):
                continue

            if "start" not in segment or "end" not in segment:
                continue

            start_idx = int(float(segment["start"]))
            end_idx = int(float(segment["end"]))

            start_idx = max(0, min(start_idx, wav_len))
            end_idx = max(0, min(end_idx, wav_len))

            if end_idx <= start_idx:
                continue

            segments.append(wav[start_idx:end_idx])

        return segments