import os

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Inference, Model
from pyannote.audio.utils.powerset import Powerset
import torchaudio

from lib.models.module import DiarizationModule
from lib.const import SAMPLING_RATE

from silero_vad import read_audio

load_dotenv()


class PyannoteOverlapModule(DiarizationModule):
    """
    Detect overlap in individual VAD segments using 10-second sliding windows.

    Input VAD segments are expected in sample indices:
    [{"start": int, "end": int}, ...]
    """

    def __init__(
        self,
        audio_path: str,
        vad_segments: list[dict],
        threshold: float = 0.5,
        model_name: str = "pyannote/segmentation-3.0",
    ):
        super().__init__(tag="Pyannote Overlap Detection")
        self.audio_path = audio_path
        self.vad_segments = vad_segments
        self.threshold = float(threshold)
        self.model_name = model_name
        model = Model.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        self.model = model
        self.inference = Inference(model, window="whole")

        specs = model.specifications
        num_classes = specs.num_classes if hasattr(specs, "num_classes") else 4
        max_set_size = specs.max_set_size if hasattr(specs, "max_set_size") else 2
        
        self._to_multilabel = Powerset(num_classes, max_set_size).to_multilabel
        self.max_speakers_per_chunk = num_classes
        self.max_speakers_per_frame = max_set_size

    @staticmethod
    def _remove_short_runs(mask: np.ndarray, min_frames: int = 5) -> np.ndarray:
        if mask.size == 0:
            return mask

        x = mask.astype(np.int8)
        diff = np.diff(np.concatenate(([0], x, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        out = np.zeros_like(mask, dtype=bool)
        for s, e in zip(starts, ends):
            if (e - s) >= min_frames:
                out[s:e] = True
        return out

    def _detect_overlap_in_segment(self, segment_start: int, segment_end: int) -> list[tuple[int, int]]:
        """
        Detect overlap intervals within a single VAD segment.
        
        Args:
            segment_start: start sample index
            segment_end: end sample index
            
        Returns:
            List of (start_sample, end_sample) tuples for overlap regions within this segment
        """
        wav, _= torchaudio.load(self.audio_path)
        
        segment_wav = wav[:, segment_start:segment_end]
        print(segment_wav)
        
        try:
            output = self.inference({"waveform": segment_wav, "sample_rate": SAMPLING_RATE})
        except Exception as e:
            print(f"[ERROR] Inference failed for segment [{segment_start}, {segment_end}]: {e}")
            return []
        
        scores_t = torch.from_numpy(np.asarray(output.data, dtype=np.float32))
        
        try:
            scores_multilabel = self._to_multilabel(scores_t)
            scores = scores_multilabel.detach().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Powerset decode failed: {e}")
            return []

        hot_positions = np.argmax(scores.flatten().reshape(scores.shape[0], -1), axis=-1)  
        
        position_changes = np.diff(hot_positions.astype(int))
        change_frames = np.where(position_changes != 0)[0] + 1  
        
        print(f"[DEBUG] Speaker changes detected: {len(change_frames)} frames")
        
        if len(change_frames) == 0:
            return []
        
        num_frames = scores.shape[0] if scores.ndim >= 1 else 1
        frame_duration = (segment_end - segment_start) / num_frames / SAMPLING_RATE
        
        transition_points: list[tuple[int, int]] = []
        
        for frame_idx in change_frames:
            if frame_idx >= num_frames:
                continue
            
            frame_start_time = frame_idx * frame_duration
            frame_end_time = (frame_idx + 1) * frame_duration
            
            s_sample = segment_start + int(frame_start_time * SAMPLING_RATE)
            e_sample = segment_start + int(frame_end_time * SAMPLING_RATE)
            transition_points.append((s_sample, e_sample))
        
        print(f"[DEBUG] Returning {len(transition_points)} transition points")
        return transition_points


    @staticmethod
    def _split_segment_by_overlaps(
        start: int,
        end: int,
        overlap_intervals: list[tuple[int, int]],
    ) -> list[dict]:
        pieces = [{"start": start, "end": end, "is_overlap": False}]

        for ov_start, ov_end in overlap_intervals:
            new_pieces = []
            for piece in pieces:
                ps, pe, is_overlap = piece["start"], piece["end"], piece["is_overlap"]
                if pe <= ov_start or ps >= ov_end:
                    new_pieces.append(piece)
                    continue

                inter_s = max(ps, ov_start)
                inter_e = min(pe, ov_end)

                if ps < inter_s:
                    new_pieces.append({"start": ps, "end": inter_s, "is_overlap": is_overlap})
                if inter_s < inter_e:
                    new_pieces.append({"start": inter_s, "end": inter_e, "is_overlap": True})
                if inter_e < pe:
                    new_pieces.append({"start": inter_e, "end": pe, "is_overlap": is_overlap})

            pieces = new_pieces

        return [p for p in pieces if p["end"] > p["start"]]

    def run(self):
        """
        Process each VAD segment independently for overlap detection.
        """
        print(f"[DEBUG] Processing {len(self.vad_segments)} VAD segments for overlap detection")
        
        all_overlap_intervals: list[tuple[int, int]] = []
        
        for seg_idx, seg in enumerate(self.vad_segments):
            if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
                continue
            
            start_sample = int(seg["start"])
            end_sample = int(seg["end"])
            segment_duration_sec = (end_sample - start_sample) / SAMPLING_RATE
            
            print(f"[DEBUG] Segment {seg_idx+1}/{len(self.vad_segments)}: [{start_sample}, {end_sample}] ({segment_duration_sec:.2f}s)")
            
            overlap_intervals = self._detect_overlap_in_segment(start_sample, end_sample)
            all_overlap_intervals.extend(overlap_intervals)
            
            if overlap_intervals:
                print(f"  → Found {len(overlap_intervals)} overlap regions")
        
        print(f"[DEBUG] Total overlaps found: {len(all_overlap_intervals)}")
        
        split_vad_segments = []
        for seg in self.vad_segments:
            if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
                continue
            split_vad_segments.extend(
                self._split_segment_by_overlaps(
                    int(seg["start"]),
                    int(seg["end"]),
                    all_overlap_intervals,
                )
            )

        new_vad_segments = sorted(
            [{"start": s["start"], "end": s["end"]} for s in split_vad_segments 
             if not s.get("is_overlap", False)],
            key=lambda x: x["start"],
        )

        print(f"[DEBUG] Returning {len(new_vad_segments)} non-overlap segments")
        return new_vad_segments