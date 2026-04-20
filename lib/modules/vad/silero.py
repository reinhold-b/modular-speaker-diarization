from lib.models.module import DiarizationModule
from silero_vad import (load_silero_vad, 
                        VADIterator, 
                        read_audio, 
                        get_speech_timestamps)

from lib.const import *

class SileroVADModule(DiarizationModule):
    def __init__(self):
        super().__init__(tag="Silero VAD")
        self.silero_vad_model = load_silero_vad()

    def run(self, audio_path: str):
        wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav, 
            self.silero_vad_model, 
            sampling_rate=SAMPLING_RATE,  
            threshold=0.5,                
            min_speech_duration_ms=250,    
            #max_speech_duration_s=3,    
            min_silence_duration_ms=500,   
            speech_pad_ms=80,
        )
        return speech_timestamps