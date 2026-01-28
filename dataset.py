import pandas as pd
import numpy as np
import torch
import re

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter


class IMDBDataset(Dataset):
    def __init__(self, sequences, labels):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IMDBDataModule:
    def __init__(
        self,
        csv_path,
        text_col="review",
        label_col="sentiment",
        num_words=10000,
        maxlen=200,
        batch_size=32,
        test_size=0.2
    ):
        self.csv_path = csv_path
        self.text_col = text_col
        self.label_col = label_col
        self.num_words = num_words
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.test_size = test_size

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

    def setup(self):
        df = pd.read_csv(self.csv_path)

        texts = df[self.text_col].astype(str).values
        labels = df[self.label_col].apply(
            lambda x: 1 if x == "positive" else 0
        ).values

        tokenized = [self._clean_text(t) for t in texts]

        counter = Counter()
        for tokens in tokenized:
            counter.update(tokens)

        self.vocab = {"<PAD>": 0, "<OOV>": 1}
        for i, (word, _) in enumerate(
            counter.most_common(self.num_words - 2), start=2
        ):
            self.vocab[word] = i

        def to_seq(tokens):
            seq = [self.vocab.get(tok, 1) for tok in tokens]
            if len(seq) < self.maxlen:
                seq += [0] * (self.maxlen - len(seq))
            else:
                seq = seq[:self.maxlen]
            return seq

        sequences = np.array([to_seq(t) for t in tokenized])

        X_train, X_test, y_train, y_test = train_test_split(
            sequences,
            labels,
            test_size=self.test_size,
            random_state=42
        )

        self.train_loader = DataLoader(
            IMDBDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            IMDBDataset(X_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

    @property
    def vocab_size(self):
        return len(self.vocab)
