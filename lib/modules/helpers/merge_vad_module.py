from lib.models.module import DiarizationModule
from lib.const import SAMPLING_RATE

class MergeVADSegmentsModule(DiarizationModule):
    """
    Merge VAD segments that are close to each other.
    """

    def __init__(self, vad_segments: list[tuple[float, float]], merge_threshold: float = 1):
        super().__init__(tag="Merge VAD Segments")
        self.vad_segments = vad_segments
        self.merge_threshold = merge_threshold

    def run(self):

        def _to_seconds(segment):
            if isinstance(segment, dict) and "start" in segment and "end" in segment:
                start_sec = float(segment["start"]) / SAMPLING_RATE
                end_sec = float(segment["end"]) / SAMPLING_RATE
                return {"start": start_sec, "end": end_sec}
            return None

        def _to_samples(segment):
            if isinstance(segment, dict) and "start" in segment and "end" in segment:
                start_sample = int(float(segment["start"]) * SAMPLING_RATE)
                end_sample = int(float(segment["end"]) * SAMPLING_RATE)
                return {"start": start_sample, "end": end_sample}
            return None
        
        if not self.vad_segments:
            return []

        merged_segments = []
        print(self.vad_segments)
        current_start = _to_seconds(self.vad_segments[0])["start"]
        current_end = _to_seconds(self.vad_segments[0])["end"]

        for segment in self.vad_segments[1:]:
            start = _to_seconds(segment)["start"]
            end = _to_seconds(segment)["end"]
            print(start - current_end)
            if start - current_end <= self.merge_threshold:
                current_end = max(current_end, end)
            else:
                merged_segments.append(_to_samples({"start": current_start, "end": current_end}))
                current_start, current_end = start, end

        merged_segments.append(_to_samples({"start": current_start, "end": current_end}))
        return merged_segments