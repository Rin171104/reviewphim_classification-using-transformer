# IMDB Movie Review Classification with Transformer

This project implements a Transformer-based model for sentiment classification (Positive/Negative) on the IMDB Movie Review dataset using PyTorch.

## Project Structure

- `dataset.py`: Handles data loading, preprocessing, and `IMDBDataModule` preparation.
- `model.py`: Defines the Transformer model architecture including `PositionalEncoding`, `TransformerEncoderBlock`, and `TransformerClassifier`.
- `train.py`: Main script for training the model, evaluating, and saving checkpoints.
- `requirements.txt`: List of Python dependencies.

## Requirements

Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The project expects the IMDB Dataset in CSV format (`IMDB Dataset.csv`). Ensure you have the dataset available. The default path in `train.py` might need to be adjusted or passed as an argument if it's not in the expected location.

## Usage

### Training the Model

You can start training by running `train.py`. You can adjust hyperparameters using command-line arguments.

```bash
python train.py --csv-path "IMDB Dataset.csv" --epochs 20 --batch-size 64
```

**Arguments:**

- `--csv-path`: Path to the IMDB dataset CSV file.
- `--epochs`: Number of training epochs (default: 20).
- `--batch-size`: Batch size for training (default: 64).
- `--num-words`: Vocabulary size (default: 500).
- `--maxlen`: Maximum sequence length (default: 500).
- `--logging`: Directory for TensorBoard logs (default: "tensorboard").
- `--trained-models`: Directory to save trained models (default: "trained-models").
- `--checkpoint`: Path to a checkpoint to resume training from (optional).

## Logging

Training progress and metrics are logged using TensorBoard. You can view the logs by running:

```bash
tensorboard --logdir tensorboard
```

## Model Architecture

The model consists of:
1.  **Embedding Layer**: Converts token indices to dense vectors.
2.  **Positional Encoding**: Adds positional information to the embeddings.
3.  **Transformer Encoder Blocks**: Multi-head attention and feed-forward networks with normalization and dropout.
4.  **Global Average Pooling**: Aggregates the sequence of vectors.
5.  **Classifier Head**: Fully connected layers to predict the sentiment.
