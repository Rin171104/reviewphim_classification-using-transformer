import torch
import torch.nn as nn
import os
import shutil
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from dataset import IMDBDataModule
from model import TransformerClassifier


def get_args():
    parser = ArgumentParser("IMDB LSTM Training")
    parser.add_argument(
        "--csv-path", "-d",
        type=str,
        default=r"D:/HocPython/deep learning/NLP/reviewphim_classification/IMDB Dataset.csv",
        help="Path to IMDB CSV file"
    )
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--num-words", type=int, default=500)
    parser.add_argument("--maxlen", type=int, default=500)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained-models", "-t", type=str, default="trained-models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # ======================
    # TensorBoard reset
    # ======================
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    writer = SummaryWriter(args.logging)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    # ======================
    # Dataset
    # ======================
    data_module = IMDBDataModule(
        csv_path=args.csv_path,
        num_words=args.num_words,
        maxlen=args.maxlen,
        batch_size=args.batch_size
    )
    data_module.setup()

    train_loader = data_module.train_loader
    test_loader = data_module.test_loader
    num_iter = len(train_loader)

    # ======================
    # Device
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======================
    # Model
    # ======================
    model = TransformerClassifier(
        vocab_size=data_module.vocab_size,
        maxlen=args.maxlen
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ======================
    # Load checkpoint
    # ======================
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        print(f"Resumed from epoch {start_epoch}, best acc = {best_accuracy:.4f}")
    else:
        start_epoch = 0
        best_accuracy = 0.0

    # ======================
    # Training loop
    # ======================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="cyan")
        running_loss = 0.0

        for i, (x_batch, y_batch) in enumerate(progress_bar):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch).squeeze(1)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Iter {i+1}/{num_iter} "
                f"Loss {loss.item():.4f}"
            )

            writer.add_scalar(
                "Train/Loss",
                loss.item(),
                epoch * num_iter + i
            )

        avg_loss = running_loss / num_iter
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

        # ======================
        # Evaluation
        # ======================
        model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch).squeeze(1)
                preds = (torch.sigmoid(logits) >= 0.5).long()

                all_labels.extend(y_batch.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Val Accuracy: {accuracy:.4f}")

        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        # ======================
        # Save checkpoint
        # ======================
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy
            },
            f"{args.trained_models}/last_lstm.pt"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_accuracy": best_accuracy
                },
                f"{args.trained_models}/best_lstm.pt"
            )

        torch.cuda.empty_cache()
