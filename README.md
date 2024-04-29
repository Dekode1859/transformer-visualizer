# Transformer Attention Visualizer

This repository contains an implementation of the Transformer model in PyTorch from scratch for English to Italian translation. It includes training scripts, model configurations, dataset handling, inference scripts, and attention visualization.
The main purpose of this project is to visualize the famous attention mechanism in transformer models, and for that the best way to do so would be implementing everything from scratch in PyTorch.

## Repository Structure

- `app.py`: Main Streamlit script for running the web application deployed on [Hugging Face Spaces](https://dekode-transformer-visualizer.hf.space).
- `config.py`: Configuration file containing model configurations and hyperparameters for the model used during training and inference.
- `dataset.py`: Script for handling datasets, including loading and preprocessing.
- `model.py`: Implementation of the Transformer model in PyTorch from scratch.
- `requirements.txt`: List of Python dependencies required for the project.
- `tokenizer_en.json`: Tokenizer file for English language.
- `tokenizer_it.json`: Tokenizer file for Italian language.
- `train.py`: Training script for training the Transformer model using the [Helsinki-NLP Opus Books dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books).
- `translate.py`: Script for testing the translation functionality using a trained model.
- `attention_visual.ipynb`: Script for visualizing the attention from the model into charts and graphs.

## Tech Stack
- Python
- PyTorch
- HuggingFace datasets and tokenizers
- Pandas
- Numpy
- Streamlit
- Altair (Visualization)
- HuggingFace Spaces

## Implementations

### Model

The `model.py` file contains the implementation of the Transformer model in PyTorch. This includes the encoder, decoder, attention mechanisms, and other components necessary for translation.

### Training

The `train.py` script is used for training the Transformer model on the Helsinki-NLP Opus Books dataset, which consists of English to Italian translations. It includes training, validation, and a greedy decode loop for faster inference after training is complete. this file is used in the notebooks `Colab_Train.ipynb` and `Local_Train.ipynb` to train the model efficiently while also savings the weights and optimizer state to resume training later.

#### Config Used
```json
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
```

#### Training Details

- **Device Used**: NVIDIA GeForce RTX 3060 Laptop GPU

- **Number of Epochs Trained**: 21 epochs

- **Loss Change Over Time**:

| Epoch | Loss     |
|-------|----------|
| 0     | 6.273    |
| 1     | 5.315    |
| 2     | 4.822    |
| 3     | 4.801    |
| 4     | 3.781    |
| 5     | 3.640    |
| 6     | 4.898    |
| 7     | 3.447    |
| 8     | 4.317    |
| 9     | 2.520    |
| 10    | 3.369    |
| 11    | 4.282    |
| 12    | 3.235    |
| 13    | 2.752    |
| 14    | 2.525    |
| 15    | 2.698    |
| 16    | 1.950    |
| 17    | 3.392    |
| 18    | 2.535    |
| 19    | 2.916    |
| 20    | 2.290    |

### Configuration

The `config.py` file holds all the model configurations and hyperparameters. Users can modify these configurations according to their preferences before training.

### Inference

The `translate.py` script allows for generating translations using a trained model. It utilizes a greedy decoding strategy for inference.

### Attention Visualization

The `attention_visual.py` script is for visualizing the attention from the model into charts and graphs. This visualization is later used in the web application deployed on Hugging Face Spaces.

## Web Application

The web application deployed on Hugging Face Spaces allows users to input English sentences and visualize the attention mechanism in the Transformer model for English to Italian translation. It provides an interactive way to understand how the model translates sentences by visualizing Self-Attention for the Encoder and Decoder and Cross-Attention between the Encoder and Decoder.

[Click here to access the web application](https://dekode-transformer-visualizer.hf.space)
