"""
Convert XML reference files
to python objects to use for DER calculation.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


def xml_to_ref(filename: str):
    base_dir = Path(__file__).resolve().parents[2]
    file_path = base_dir / filename
    speaker_tag = filename.split("/")[-1].split(".")[1]
    tree = ET.parse(file_path)
    root = tree.getroot()

    ref = []
    for segment in root.findall("segment"):
        start = segment.attrib.get("transcriber_start")
        end = segment.attrib.get("transcriber_end")
        if start is not None and end is not None:
            ref.append((speaker_tag, float(start), float(end)))

    return ref

def load_refs_from_audio_file(audio_file: str):
    base_dir = Path(__file__).resolve().parents[2]
    print(base_dir)
    file_path = base_dir.joinpath("datasets/amicorpus/ami_public_manual_1.6.2/segments")
    keyword = audio_file.split("/")[-1].split(".")[0]
    print(file_path)
    print(keyword)
    refs = []
    for file in file_path.rglob(f"*{keyword}*"):
        logger.info(f"Processing file: {file}")
        ref = xml_to_ref(str(file))
        refs.extend(ref)

    refs.sort(key=lambda x: float(x[1])) 
    return refs

def get_rttm_from_audio_file(audio_file: str) -> str:
    """
    Loads reference segments for an audio file and returns them in RTTM format.
    Format: SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    refs = load_refs_from_audio_file(audio_file)
    file_id = audio_file.split("/")[-1].split(".")[0]
    
    rttm_lines = []
    for speaker_tag, start, end in refs:
        duration = float(end) - float(start)
        # RTTM Standard: SPEAKER file_id 1 turn_onset turn_duration <NA> <NA> speaker_name <NA> <NA>
        line = f"SPEAKER {file_id} 1 {float(start):.3f} {duration:.3f} <NA> <NA> {speaker_tag} <NA> <NA>"
        rttm_lines.append(line)
        
    return "\n".join(rttm_lines)

if __name__ == "__main__":
    pass