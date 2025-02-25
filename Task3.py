import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
# task 1
def load_vocab(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]
    w2i = {word: idx for idx, word in enumerate(vocab)}
    i2w = {idx: word for word, idx in w2i.items()}
    return w2i, i2w

w2i, i2w = load_vocab("vocabulary_2.txt")
vocab_size = len(w2i)
# task 2 load
word2vec_model = torch.load("word2vec_model.pth", map_location="cpu")
embedding_weights = word2vec_model["embeddings.weight"].clone().detach()

if embedding_weights.shape[0] != vocab_size:
    print(f" adjusting embeddings: {embedding_weights.shape[0]} → {vocab_size}")
    new_embedding_matrix = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_weights.shape[1]))
    new_embedding_matrix[:embedding_weights.shape[0], :] = embedding_weights.cpu().numpy()
    embedding_weights = torch.tensor(new_embedding_matrix, dtype=torch.float)

print(f"Vocabulary Size: {vocab_size}, Embedding Dimension: {embedding_weights.shape[1]}")

# dataset
class NeuralLMDataset(Dataset):
    def __init__(self, corpus_file, w2i, context_size=3):
        self.context_size = context_size
        self.w2i = w2i
        self.unk_idx = w2i.get("<UNK>", len(w2i))  # Assign <UNK> index
        self.data = self.process_data(corpus_file)

    def process_data(self, corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f.readlines()]
        
        data = []
        for sentence in sentences:
            tokens = sentence.split()
            token_indices = [self.w2i.get(token, self.unk_idx) for token in tokens]
            
            for i in range(len(token_indices) - self.context_size):
                context = token_indices[i:i + self.context_size]
                target = token_indices[i + self.context_size]
                data.append((context, target))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Load dataset
tr_ds = NeuralLMDataset("corpus.txt", w2i)
train_size = int(0.8 * len(tr_ds))
valid_size = len(tr_ds) - train_size
train_subset, val_sub = random_split(tr_ds, [train_size, valid_size])
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
valid_loader = DataLoader(val_sub, batch_size=64, shuffle=False)
print(f"Dataset Size: {len(tr_ds)}")
#perplexity from loss
def calculate_perplexity(loss):
    return np.exp(loss)

# archet...
class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, dropout=0.6):
        # simple mlp
        super(NeuralLM1, self).__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding.from_pretrained(embedding_weights.clone().detach(), freeze=False)
        self.fc = nn.Linear(embedding_dim * self.context_size, vocab_size)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], self.context_size * self.embedding.embedding_dim)
        x = self.dropout(x)
        return self.fc(x)
#  deeper mlp
class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NeuralLM2, self).__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding.from_pretrained(embedding_weights.clone().detach(), freeze=False)
        self.fc1 = nn.Linear(embedding_dim * self.context_size, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], self.context_size * self.embedding.embedding_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
# lstm 
class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128, num_layers=2):
        super(NeuralLM3, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights.clone().detach(), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# training 
def train(model, train_loader, valid_loader, epochs=10, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_loader:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for context, target in valid_loader:
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

    return train_losses, valid_losses

# evaluate accuracy and perplexity 
def evaluate_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for context, target in data_loader:
            context, target = context.to(device), target.to(device)
            output = model(context)

            loss = criterion(output, target)
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=1)
            total_correct += (predictions == target).sum().item()
            total_samples += target.size(0)

    accuracy = total_correct / total_samples
    perplexity = calculate_perplexity(total_loss / len(data_loader))

    return accuracy, perplexity
#predict next three tokens
def pred_next(model, sentence, num_tokens=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    tokens = sentence.split()
    token_indices = [w2i[token] for token in tokens if token in w2i]
    
    if len(token_indices) < 3:
        raise ValueError("Input sentence must have at least 3 tokens")
    
    context = token_indices[-3:]  # Take the last 3 tokens as context
    context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_tokens):
            output = model(context_tensor)
            next_token_idx = torch.argmax(output, dim=1).item()
            predictions.append(i2w[next_token_idx])
            context.append(next_token_idx)
            context = context[1:]  # Shift context window
            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    return predictions
# Run prediction on test sentences
def run_predictions(model, test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    
    for sentence in sentences:
        print(f"Input: {sentence}")
        try:
            predictions = pred_next(model, sentence)
            print(f"Predicted next words: {' '.join(predictions)}")
        except ValueError as e:
            print(f"Skipping sentence: {e}")
        print("-")
# Trains
models = {
    "NeuralLM1": NeuralLM1(vocab_size, 300, 3),
    "NeuralLM2": NeuralLM2(vocab_size, 300, 3),
    "NeuralLM3": NeuralLM3(vocab_size, 300, 3)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    train_loss, valid_loss = train(model, train_loader, valid_loader, epochs=10)
    torch.save(model.state_dict(), f"{name}.pth")
    
    train_acc, train_perp = evaluate_model(model, train_loader)
    valid_acc, valid_perp = evaluate_model(model, valid_loader)
    
    print(f"\n{name} - Train Accuracy: {train_acc:.4f}, Train Perplexity: {train_perp:.4f}")
    print(f"{name} - Valid Accuracy: {valid_acc:.4f}, Valid Perplexity: {valid_perp:.4f}")
    
    plt.plot(train_loss, label=f"{name} - Train Loss")
    plt.plot(valid_loss, label=f"{name} - Valid Loss", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{name} - Training vs Validation Loss")
    plt.show()
# testing file
test_file = "sample_test.txt"
for name, model in models.items():
    print(f"\nRunning predictions for {name}...")
    run_predictions(model, test_file) 
