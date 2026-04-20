import json
import logging
import numpy as np
import sounddevice as sd
from silero_vad import (load_silero_vad, 
                        VADIterator, 
                        read_audio, 
                        get_speech_timestamps)
import torch
import os
import argparse 
import torchaudio
import tqdm
from lib.const import *

if not hasattr(torchaudio, "list_audio_backends"):
    print("here")
    torchaudio.list_audio_backends = lambda: [""] 

from lib.log import get_system_logger
from lib.vad_data_loader import VADDataLoader
from dotenv import load_dotenv
from pyannote.audio import Inference, Model
load_dotenv()

SAMPLING_RATE = 16_000
VAD_WINDOW_SIZE = 512


# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = get_system_logger()
silero_vad_model = load_silero_vad()
silero_iterator = VADIterator(silero_vad_model, sampling_rate=SAMPLING_RATE)

def extract_embedding(speaker_audio):
    """
    Use the VAD audio chunks to create speaker embeddings.
    """
    if speaker_audio is None or len(speaker_audio) == 0:
        return None

    audio_tensor = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)
    embedding = inference({
        "waveform": audio_tensor,
        "sample_rate": SAMPLING_RATE,
    })

    if torch.is_tensor(embedding):
        embedding = embedding.squeeze().detach().cpu().numpy()
    else:
        embedding = np.asarray(embedding).squeeze()

    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    with open("data.csv", "a+") as f:
        np.savetxt(f, embedding.reshape(1, -1), delimiter=",")

    return embedding

def live_system():
    logger.info("Live diarization system started.")
    # 512 samples at 16 kHz = 32 ms latency per VAD decision
    chunk_size = VAD_WINDOW_SIZE
    audio_buffer = np.array([], dtype=np.float32)
    is_speaking = False
    current_speech_chunks = []

    def callback(indata, frames, time, status):
        nonlocal audio_buffer, is_speaking, current_speech_chunks

        if status:
            logger.warning(f"Audio callback status: {status}")

        audio_chunk = indata[:, 0].astype(np.float32)
        audio_buffer = np.concatenate((audio_buffer, audio_chunk))

        while len(audio_buffer) >= VAD_WINDOW_SIZE:
            chunk = audio_buffer[:VAD_WINDOW_SIZE]
            audio_buffer = audio_buffer[VAD_WINDOW_SIZE:]

            speech_dict = silero_iterator(chunk, return_seconds=True)

            if is_speaking:
                current_speech_chunks.append(chunk)

            if speech_dict and speech_dict.get("start") and not is_speaking:
                is_speaking = True
                current_speech_chunks = [chunk]
                logger.info("Speaking")
            elif speech_dict and speech_dict.get("end") and is_speaking:
                is_speaking = False
                logger.info("Stopped speaking.")
                speech_audio = np.concatenate(current_speech_chunks) if current_speech_chunks else np.array([], dtype=np.float32)
                extract_embedding(speech_audio)
                current_speech_chunks = []


    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        callback=callback,
        blocksize=chunk_size
    ):
        logger.info("Recording started. Speak now (Ctrl+C to stop)")
        while True:
            sd.sleep(50)

def main(audio_path: str):
    logger.info("Dataset diarization system started.")
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)

    logger.info("Starting timestamo extraction.")
    speech_timestamps = get_speech_timestamps(
        wav,
        silero_vad_model,
        return_seconds=True, 
        sampling_rate=SAMPLING_RATE, 
        threshold=0.35,                
        min_speech_duration_ms=300,    
        max_speech_duration_s=2.5,    
        min_silence_duration_ms=500,   
        speech_pad_ms=100,
    )

    with open("seconds.json", "w+") as file:
        file.write(json.dumps(speech_timestamps))
        file.close()

    logger.info("Extracted segments from audio and written to file.")

    wav_segments = VADDataLoader.load_from_json("seconds.json", "")

    logger.info("Loaded VAD segments from audio.")
    logger.info("Starting embedding extraction.")
    
    for segment in tqdm.tqdm(wav_segments):
        extract_embedding(segment)

    logger.info("Written embeddings to file.")




parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to an audio file.", type=str, required=False)

if __name__ == "__main__":
    logger.debug("Starting...")
    
    args = parser.parse_args()

    path = args.path

    try:
        if path:
            main(path)
        else:
            live_system()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        silero_iterator.reset_states()
