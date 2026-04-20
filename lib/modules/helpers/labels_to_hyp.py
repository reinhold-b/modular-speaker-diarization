import numpy as np
from lib.const import SAMPLING_RATE

class LabelsToHypModule:
    def __init__(self, labels: np.ndarray, vad_segments: list, refs: list = None):
        self.labels = labels
        self.vad_segments = vad_segments
        self.refs = refs
    
    def _build_hyp_with_cluster_ids(self) -> list:
        """Build hypothesis with numeric cluster IDs first"""
        hyp_clusters = []
        for i, label in enumerate(self.labels):
            if i >= len(self.vad_segments):
                break
            
            seg = self.vad_segments[i]
            if "start" in seg and isinstance(seg["start"], int) and seg["start"] > 10000: 
                start_time = seg["start"] / SAMPLING_RATE
                end_time = seg["end"] / SAMPLING_RATE
            elif "start" in seg:
                start_time = float(seg["start"])
                end_time = float(seg["end"])
            else:
                start_time = 0.0
                end_time = 0.0
                
            hyp_clusters.append((int(label), start_time, end_time))
        
        return hyp_clusters
    
    def execute(self):
        hyp_clusters = self._build_hyp_with_cluster_ids()
        
        hyp = []
        for cluster_id, start_time, end_time in hyp_clusters:
            hyp.append((f"spk_{cluster_id}", start_time, end_time))
        
        hyp.sort(key=lambda x: x[1])
        return hyp