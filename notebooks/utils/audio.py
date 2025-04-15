import os
from typing import Dict, Any

import librosa
import torch
import pandas as pd

from transformers import Wav2Vec2Processor, Wav2Vec2Model

from zipfile import ZipFile
from datasets import Dataset, concatenate_datasets, load_dataset
from typing import List

from pydub import AudioSegment

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
labels = pd.read_json("./../data/ludwig-music-dataset-moods-and-subgenres/labels.json")


def download_kaggle_dataset(data_set_path: str) -> None:
    name = data_set_path.split("/")[1]

    result = os.system(f"kaggle datasets download -d {data_set_path}")

    print(result)

    with ZipFile(f"./{name}.zip") as zip_ref:
        zip_ref.extractall(f"../data/{name}")

    os.remove(f"./{name}.zip")


def load_sample_from_genre(genre: str, n_samples: int) -> Dataset:
    data_path = f"./../data/ludwig-music-dataset-moods-and-subgenres/mp3/mp3/{genre}"
    data_set = (
        load_dataset(cache_dir="./../data/audiofolder", path=data_path, split="train")
        .shuffle(42)
        .select(range(n_samples))
    )

    return data_set


def generate_random_dataset(genres: List[str], n_samples: int) -> Dataset:
    ds = None
    for genre in genres:
        print(f"Loading {genre}...")
        if ds is None:
            ds = load_sample_from_genre(genre, n_samples)
        else:
            ds_aux = load_sample_from_genre(genre, n_samples)
            ds = concatenate_datasets([ds_aux, ds])

    del ds_aux
    return ds


def convert_mp3_to_wav(path: str, output_path: str) -> None:
    """Convert an MP3 file to a WAV file (Signed 16-bit PCM)."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    sound = AudioSegment.from_mp3(path)

    # Ensure the output directory exists
    dir_path = os.path.dirname(output_path)
    os.makedirs(dir_path, exist_ok=True)

    # Export to WAV (16-bit PCM)
    sound.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

    print(f"Converted {path} to {output_path}")


def load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load an audio file and return it as a tensor."""
    waveform, _ = librosa.load(path, sr=sample_rate)

    print(f"Loaded {path} with shape {waveform.shape}")
    return waveform


def extract_embeddings(path: str, model_name: str) -> torch.Tensor:
    """Extract embeddings from an audio file using Wav2Vec2."""
    print(f"Extracting embeddings from {path}...")
    print(f"With device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name, ignore_mismatched_sizes=True)

    audio_tensor = load_audio(path)
    print(f"Audio tensor shape: {audio_tensor.shape}")

    inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_audio_data(identifier: str, genre: str, model_name: str) -> Dict[str, Any]:
    """Get audio data from a file."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(
        base_path,
        "../../data/ludwig-music-dataset-moods-and-subgenres/mp3/mp3",
        genre,
        f"{identifier}.mp3",
    )
    audio_tensor = load_audio(audio_path)

    embeddings = extract_embeddings(audio_path, model_name)

    return {
        "audio": audio_tensor,
        "embeddings": embeddings,
        "metadata": labels["tracks"][identifier],
        "identifier": identifier,
    }
