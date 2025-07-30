import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import TextCNN
from data import TextDataset
from utils.utils import *
from gensim.models import KeyedVectors

def train(model, device, train_loader, optimizer):
  model.train()
  correct = 0
  total = 0

  train_loss = 0.0

  for batch_idx, (text, label) in enumerate(train_loader):
    text, label = text.to(device), label.to(device)

    optimizer.zero_grad()
    output = model(text)
    loss = F.cross_entropy(output, label)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

    pred = output.argmax(dim=1, keepdim=True).squeeze()
    correct += (pred == label).sum().item()
    total += label.size(0)

  avg_loss = train_loss / len(train_loader)
  accuracy = correct / total * 100

  return avg_loss, accuracy

def evaluate(model, device, valid_loaber, optimizer):
  model.eval()

  valid_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, (text, label) in enumerate(valid_loader):
      text, label = text.to(device), label.to(device)

      output = model(text)
      loss = F.cross_entropy(output, label)

      valid_loss += loss.item()


      pred = output.argmax(dim=1, keepdim=True).squeeze()
      correct += (pred == label).sum().item()
      total += label.size(0)

  avg_loss = valid_loss / len(valid_loader)
  accuracy = correct / total * 100

  return avg_loss, accuracy

if __name__ == '__main__':
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data, valid_data, _, all_tokens = load_and_prepare_data()
    
    vocab = build_vocab(all_tokens)
    word2vec = KeyedVectors.load_word2vec_format(config["word2vec_path"], binary=True)
    embedding_matrix = torch.rand(len(vocab), config["embedding_dim"]).uniform_(-0.25, 0.25)
    
    for word, idx in vocab.items():
        if word in word2vec:
            embedding_matrix[idx] = torch.tensor(word2vec[word])
    
    # Dataset and Dataloader
    train_dataset = TextDataset(train_data, vocab, config["max_len"], encode)
    valid_dataset = TextDataset(valid_data, vocab, config["max_len"], encode)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Model
    model = TextCNN(embedding_matrix, config["out_channels"], config["num_classes"], config["dropout"])
    model = model.to(device)
    
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train(model, device, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, device, valid_loader, optimizer)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"          Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print("Early stopping triggered.")
            break
