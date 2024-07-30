import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn, optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize

# Load the datasets
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

# Define constants
MAX_SEQUENCE_LENGTH = 200  # Maximum length of sequences
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 100
HIDDEN_DIM = 128

# Tokenize the text
def tokenize(text):
    return word_tokenize(text.lower())

# Build the vocabulary
def build_vocab(data):
    counter = Counter()
    for text in data:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

vocab = build_vocab(pd.concat([train_data['text'], val_data['text']]))

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        print()
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in self.tokenizer(text)]
        tokens = tokens[:self.max_len]  # Truncate to max length
        tokens = tokens + [self.vocab['<pad>']] * (self.max_len - len(tokens))  # Pad sequences    
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Prepare datasets and dataloaders
def create_data_loader(data, vocab, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=data['text'].to_numpy(),
        labels=data['label'].to_numpy(),
        vocab=vocab,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


train_data_loader = create_data_loader(train_data, vocab, tokenize, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
val_data_loader = create_data_loader(val_data, vocab, tokenize, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, vocab, tokenize, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
print(train_data_loader)
# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)  # Adding dropout for regularization

    def forward(self, x):
        embedded = self.embedding(x)  # Convert token indices to embeddings
        _, hidden = self.gru(embedded)  # Get the hidden state from GRU
        hidden = self.dropout(hidden)  # Apply dropout to the hidden state
        output = self.fc(hidden.squeeze(0))  # Pass the hidden state through the fully connected layer
        return output

# Model instance
model = GRUModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, len(train_data['label'].unique()), vocab["<pad>"])

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and evaluation functions
def train_epoch(model, data_loader, criterion, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for i, batch in enumerate(tqdm(data_loader)):
        
        tokens = batch['tokens'].to(device)
        labels = batch['label'].to(device)

        outputs = model(tokens)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            tokens = batch['tokens'].to(device)
            labels = batch['label'].to(device)

            outputs = model(tokens)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

if __name__ == '__main__':
    # Training loop
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
    
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            criterion,
            optimizer,
            device
        )
    
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            criterion,
            device
        )
    
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Val   loss {val_loss} accuracy {val_acc}')
    
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
    
    # Evaluate on test set
    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        criterion,
        device
    )
    print(f'Test Accuracy: {test_acc * 100:.2f}%')