# Audio App: Music Dataset Vectorization and Hybrid Search

## Overview

This project provides a complete pipeline for working with a large-scale music dataset, including:
- Downloading and preparing audio data and metadata
- Converting audio files to a standard format
- Extracting audio embeddings using state-of-the-art models (e.g., Wav2Vec2, PANNs)
- Storing vectors and metadata in vector databases (Qdrant, Elasticsearch)
- Performing hybrid search (vector + metadata) for music retrieval
- Interactive exploration and analysis in Jupyter Notebooks

## Features
- **Dataset Download**: Fetches music datasets from Kaggle, including genres, moods, and subgenres.
- **Audio Preprocessing**: Converts MP3 files to WAV, loads audio, and handles missing/corrupt files.
- **Embedding Extraction**: Uses models like Facebook's Wav2Vec2 and PANNs for audio feature extraction.
- **Metadata Handling**: Loads and processes song metadata (artist, genre, mood, popularity, etc.).
- **Vector Database Integration**: Supports Qdrant and Elasticsearch for storing and searching embeddings.
- **Hybrid Search**: Combines vector similarity and metadata filters for powerful music search.
- **Jupyter Notebooks**: Step-by-step notebooks for data preparation, embedding, and search workflows.

## Project Structure

```
├── data/                # Downloaded datasets, audio files, and features
├── notebooks/           # Jupyter notebooks for all workflows
│   ├── audio.ipynb      # Qdrant vectorization and search
│   ├── audio_v3.ipynb   # Embedding extraction and Elasticsearch demo
│   └── workshop.ipynb   # End-to-end hybrid search workshop
├── src/
│   └── audio_app/       # (Optional) Python package code
├── utils/
│   └── audio.py         # Audio processing and embedding utilities
├── pyproject.toml       # Poetry project configuration
├── docker-compose.yml   # (Optional) For running Qdrant/Elasticsearch locally
└── README.md            # Project documentation
```

## Setup

### 1. Clone the Repository
```sh
git clone <repo-url>
cd audio_app
```

### 2. Install Dependencies
This project uses [Poetry](https://python-poetry.org/) for dependency management.

```sh
poetry install
```

### 3. Download the Dataset
You need a Kaggle account and API key. Place your `kaggle.json` in the project root.

```sh
poetry run python -c "from utils.audio import download_kaggle_dataset; download_kaggle_dataset('jorgeruizdev/ludwig-music-dataset-moods-and-subgenres')"
```

### 4. Start Vector Database (Optional)
- **Qdrant**: `docker-compose up qdrant`
- **Elasticsearch**: `docker-compose up elasticsearch`

### 5. Run Jupyter Notebooks
```sh
poetry run jupyter lab
```
Open any notebook in the `notebooks/` directory to follow the workflow.

## Usage

### Data Preparation
- Use `audio.ipynb` or `workshop.ipynb` to download, preprocess, and explore the dataset.

### Embedding Extraction
- Extract embeddings using Wav2Vec2 or PANNs models.
- Store embeddings and metadata in Qdrant or Elasticsearch.

### Hybrid Search
- Perform vector similarity search, metadata filtering, or hybrid queries in Elasticsearch.
- Example queries are provided in the notebooks.

## Notebooks
- **audio.ipynb**: Qdrant-based vectorization and search.
- **audio_v3.ipynb**: Embedding extraction and Elasticsearch demo.
- **workshop.ipynb**: End-to-end workflow for hybrid search.

## Utilities
- `utils/audio.py`: Functions for audio conversion, loading, embedding extraction, and dataset handling.

## Configuration
- All dependencies are managed in `pyproject.toml`.
- `.gitignore` excludes data, checkpoints, and sensitive files.

## Requirements
- Python 3.9+
- Poetry
- ffmpeg (for audio conversion)
- Docker (for running Qdrant/Elasticsearch locally)

## Troubleshooting
- Ensure `ffmpeg` is installed and available in your PATH.
- Place your Kaggle API key (`kaggle.json`) in the project root.
- If you encounter CUDA/MPS issues, check your PyTorch and hardware compatibility.

## License
This project is licensed under the Apache 2.0 License.

## Acknowledgements
- [Kaggle Ludwig Music Dataset](https://www.kaggle.com/datasets/jorgeruizdev/ludwig-music-dataset-moods-and-subgenres)
- [Qdrant](https://qdrant.tech/)
- [Elasticsearch](https://www.elastic.co/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Librosa](https://librosa.org/)

---
For questions or contributions, please open an issue or pull request.
