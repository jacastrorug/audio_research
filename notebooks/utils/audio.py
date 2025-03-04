import os

from zipfile import ZipFile
from datasets import Dataset, concatenate_datasets, load_dataset
from typing import List


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


