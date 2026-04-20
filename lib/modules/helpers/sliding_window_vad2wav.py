from lib.models.module import DiarizationModule
import numpy as np
from silero_vad import read_audio
from lib.const import SAMPLING_RATE 

class SlidingWindowVADToWavSegmentsLoader(DiarizationModule):
    def __init__(self, audio_path: str, speech_timestamps: list, sliding_window_size: float = 1.5, step: float = 0.75):
        super().__init__(tag="VAD to WAV Segments")
        self.audio_path = audio_path
        self.speech_timestamps = speech_timestamps
        self.sliding_window_size = sliding_window_size # seconds
        self.step = step # seconds
        self.window_timestamps = [] 

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

        sample_sliding_window_size = int(self.sliding_window_size * SAMPLING_RATE)
        sample_step = int(self.step * SAMPLING_RATE)

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
                
            segment_length_samples = end_idx - start_idx
            
            if segment_length_samples <= sample_sliding_window_size:
                segments.append(wav[start_idx:end_idx])
                self.window_timestamps.append({
                    "start": start_idx / SAMPLING_RATE,
                    "end": end_idx / SAMPLING_RATE
                })
                continue

            current_start_idx = start_idx
            while current_start_idx + sample_sliding_window_size <= end_idx:
                window_end_idx = current_start_idx + sample_sliding_window_size
                segments.append(wav[current_start_idx:window_end_idx])
                
                self.window_timestamps.append({
                    "start": current_start_idx / SAMPLING_RATE,
                    "end": window_end_idx / SAMPLING_RATE
                })
                
                current_start_idx += sample_step

            if current_start_idx < end_idx and (end_idx - current_start_idx) > (SAMPLING_RATE * 0.2): 
                final_window_start = max(start_idx, end_idx - sample_sliding_window_size)
                segments.append(wav[final_window_start:end_idx])
                
                self.window_timestamps.append({
                    "start": final_window_start / SAMPLING_RATE,
                    "end": end_idx / SAMPLING_RATE
                })

        return segments, self.window_timestamps