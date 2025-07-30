import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from model import TextCNN
from data import TextDataset
from utils.utils import load_and_prepare_data, encode, build_vocab, tokenizer

def evaluate(model, device, data_loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for text, label in data_loader:
            text, label = text.to(device), label.to(device)
            output = model(text)
            loss = F.cross_entropy(output, label)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    return total_loss / len(data_loader), correct / total * 100

if __name__ == "__main__":
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    print("Loading and preparing test data...")
    _, _, test_data, all_tokens = load_and_prepare_data()
    vocab = build_vocab(all_tokens)

    word2vec = KeyedVectors.load_word2vec_format(config["word2vec_path"], binary=True)
    embedding_matrix = torch.rand(len(vocab), config["embedding_dim"]).uniform_(-0.25, 0.25)

    for word, idx in vocab.items():
        if word in word2vec:
            embedding_matrix[idx] = torch.tensor(word2vec[word])

    test_dataset = TextDataset(test_data, vocab, config["max_len"], encode)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Load model
    model = TextCNN(embedding_matrix, config["out_channels"], config["num_classes"], config["dropout"])
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)

    # Evaluate
    test_loss, test_acc = evaluate(model, device, test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")