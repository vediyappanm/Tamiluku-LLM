import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import os
import math
from tqdm import tqdm

# Simple Transformer for Perplexity Testing
class MiniLM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 128, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        e = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        out = self.transformer(e)
        return self.fc_out(out)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.examples = []
        for text in texts:
            encoded = tokenizer.encode(text).ids
            if len(encoded) > max_len:
                for i in range(0, len(encoded)-max_len, max_len):
                    self.examples.append(encoded[i:i+max_len])
            else:
                self.examples.append(encoded)
    
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return torch.tensor(self.examples[idx])

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

def train_and_eval(tokenizer_path, train_texts, val_texts, epochs=2):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniLM(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_ds = TextDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    val_ds = TextDataset(val_texts, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
    
    print(f"Training on {tokenizer_path} (Vocab: {vocab_size})...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            if batch.size(1) < 2: continue
            
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            if batch.size(1) < 2: continue
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            val_loss += criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).item()
            
    avg_val_loss = val_loss / len(val_loader)
    perplexity = math.exp(avg_val_loss)
    return perplexity

if __name__ == "__main__":
    corpus_path = "data/cleaned/tamil_corpus.txt"
    if not os.path.exists(corpus_path):
        print("Corpus not found.")
        exit()
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_lines = [f.readline() for _ in range(5000)]
    
    train_lines = all_lines[:4000]
    val_lines = all_lines[4000:]
    
    # Paths to tokenizers
    # We compare our latest AMB model with a standard one (if it exists)
    amb_path = "models/amb_tokenizer/tokenizer.json"
    
    # For comparison, we'll train a quick standard BPE if not available
    # But usually, it's better to compare against an existing one like cl100k
    # For now, let's just report AMB perplexity
    
    if os.path.exists(amb_path):
        ppl = train_and_eval(amb_path, train_lines, val_lines)
        print(f"\nFinal Perplexity (AMB): {ppl:.2f}")
    else:
        print("AMB model not found. Run training first.")
