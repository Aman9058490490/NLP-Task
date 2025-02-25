import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from task1 import WordPieceTokenizer  # importing WordPiecetokenizer from task 1

# Preprocess Corpus

 #load saved vocabulary_51.txt and tokenize corpus.txt using worpieceTokenizer from 1st task

def process_corpus():
    tokenizer = WordPieceTokenizer(vocab_size=20000)
    
    # Load vocabulary from vocabulary_51.txt
    with open("vocabulary_2.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    
    word2Ind = {word: idx for idx, word in enumerate(vocab_list)}
    ind2Word = {idx: word for word, idx in word2Ind.items()}
    tokenizer.vocab = word2Ind 
    word2Ind['[PAD]'] = len(word2Ind) 
    
    print(f"processed vocabulary of {len(word2Ind)} words.")
    with open(r"C:\Users\91798\Downloads\corpus.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    tokenized_sentences = [tokenizer.tokenize(line.strip()) for line in lines]
    
    # Pad tokenized sentences to fixed length
    max_len = max(len(sentence) for sentence in tokenized_sentences)
    tokenized_sentences = [sentence + ['[PAD]'] * (max_len - len(sentence)) for sentence in tokenized_sentences]
    
    # Save tokenized corpus
    with open("tokenized_corpus.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_sentences, f, indent=4)
    
    print(f"Tokenized and padded corpus saved. Sample tokens: {tokenized_sentences[:3]}")
    return tokenized_sentences, word2Ind


# Word2Vec Dataset Class

class Word2VecDataset(Dataset):
    def __init__(self, tokenized_sentences, tokenizer, window_size=2):
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.corpus = tokenized_sentences
        self.__make_vocab()
        self.data = self.__prepare_training_data()

    def __make_vocab(self):
        word_counts = {}
        for sentence in self.corpus:
            for word in sentence:
                word_counts[word] = word_counts.get(word, 0) + 1
        self.vocab = sorted(word_counts.keys())
        self.word2Ind = {word: idx for idx, word in enumerate(self.vocab)}
        self.ind2Word = {idx: word for word, idx in self.word2Ind.items()}
        self.vocab_size = len(self.vocab)

    def __prepare_training_data(self):
        data = []
        pad_token = self.word2Ind.get('[PAD]', None)
        for sentence in self.corpus:
            for i in range(self.window_size, len(sentence) - self.window_size):
                if sentence[i] == '[PAD]':
                    continue  # Skip PAD tokens as targets
                context = [self.word2Ind[sentence[j]] for j in range(i - self.window_size, i + self.window_size + 1) if j != i and sentence[j] != '[PAD]']
                if len(context) == self.window_size * 2:  # Ensure full context window
                    target = self.word2Ind[sentence[i]]
                    data.append((context, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)



# Word2Vec CBOW Model

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(embd_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, inputs):
        embedded = self.embeddings(inputs).mean(dim=1)
        hid = self.linear1(embedded)
        out = self.linear2(hid)
        return F.log_softmax(out, dim=1)
    
    def get_embeds(self, inputs):
        inputs = inputs.to(self.embeddings.weight.device)
        return self.embeddings(inputs).view((-1, self.embeddings.embedding_dim))
    


def find_similarity_triplets(model, dataset, device):
    words = list(dataset.word2Ind.keys())[:100]
    embeddings = model.embeddings.weight.data.cpu().numpy()
    triplets = []
    
    for word in words:
        if word == "[PAD]":
            continue
        word_idx = dataset.word2Ind[word]
        word_vec = embeddings[word_idx]
        similarities = {
            other: np.dot(word_vec, embeddings[dataset.word2Ind[other]]) / 
                   (np.linalg.norm(word_vec) * np.linalg.norm(embeddings[dataset.word2Ind[other]]))
            for other in words if other != word and other != "[PAD]"
        }
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_similarities) > 2:
            triplets.append((word, sorted_similarities[0][0], sorted_similarities[-1][0]))

    print(" triplets:")
    for triplet in triplets[:2]: 
        print(f"🔹 {triplet[0]} is similar to {triplet[1]} but different from {triplet[2]}")
    
    return triplets


# Function for training 
def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs=15, patience=3):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            scores = model(context.to(device))
            loss = loss_fn(scores, target.to(device))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                scores = model(context.to(device))
                loss = loss_fn(scores, target.to(device))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "word2vec_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch+1}. Best Validation Loss: {best_val_loss:.4f}")
                break
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.legend()
    plt.show()



# function for cosine similarity 

def cosine_similarity(model, word1, word2, device):
    word1_tensor = torch.tensor([dataset.word2Ind[word1]], dtype=torch.long).to(device)
    word2_tensor = torch.tensor([dataset.word2Ind[word2]], dtype=torch.long).to(device)
    vec1 = model.get_embeds(word1_tensor).detach().cpu().numpy()
    vec2 = model.get_embeds(word2_tensor).detach().cpu().numpy()
    similarity = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity[0][0]



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenized_corpus, tokenizer = process_corpus()
    dataset = Word2VecDataset(tokenized_corpus, tokenizer, window_size=2) 
    train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)
    
    model = Word2VecModel(vocab_size=dataset.vocab_size, embd_size=512, hidden_size=128).to(device)  
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # learning rate and weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Train model
    train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs=15)

    # Load best model
    model.load_state_dict(torch.load("word2vec_model.pth"))
    model.eval()

    # Extracting words from corpus
    all_words = set([word for sentence in tokenized_corpus for word in sentence if word != '[PAD]'])
    test_words = list(all_words)
    max_pairs = 10  # limit for cosine 
    print(f"\n Cosine Similarity for {max_pairs} pairs:")
    
    count = 0
    for i in range(len(test_words)):
        for j in range(i + 1, len(test_words)):
            word1, word2 = test_words[i], test_words[j]
            if word1 in dataset.word2Ind and word2 in dataset.word2Ind:
                similarity = cosine_similarity(model, word1, word2, device)
                print(f" Cosine Similarity of ({word1}, {word2}) = {similarity:.4f}")
                count += 1
                if count >= max_pairs:
                    break
        if count >= max_pairs:
            break
    
    # for printing triplets
    print("\n Word Similarity Triplets:")
    triplets = find_similarity_triplets(model, dataset, device)
    for triplet in triplets:
        print(f"  {triplet[0]} is similar to {triplet[1]} but different from {triplet[2]}")

