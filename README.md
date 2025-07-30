# TextCNN

This project implements **TextCNN** to perform binary sentiment classification using the **Stanford Sentiment Treebank (SST-2)** dataset.  
It includes preprocessing, training, evaluation, and Word2Vec embedding integration.

## 🔧 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/TextCNN.git
cd TextCNN
```

### 2. Set up conda environment

```bash
conda create -n textcnn-env python=3.10 -y
conda activate textcnn-env
pip install -r requirements.txt
```

## 📦 Data & Embedding Preparation

### 📌 SST Dataset (Manual Download)
Download from the official website:

🔗 https://nlp.stanford.edu/sentiment/index.html

After downloading and unzipping, move the folder to:

Files required:

- datasetSentences.txt
- datasetSplit.txt
- dictionary.txt
- sentiment_labels.txt

### 📌 Word2Vec Embedding

Download from the website:

🔗 https://code.google.com/archive/p/word2vec/

Then place the file here:

`embeddings/GoogleNews-vectors-negative300.bin.gz`

## 🚀 Training the Model

```bash
python train.py
```

This will:

- Preprocess data
- Load embeddings
- Train TextCNN
- Save `best_model.pt` based on validation loss

## 📊 Evaluate the Model

```bash
python eval.py
```

This loads `best_model.pt` and prints final test accuracy.




## 🔍 Experimental Results

After training with the Stanford Sentiment Treebank (SST-2) and pre-trained GoogleNews Word2Vec embeddings, we achieved the following results on the test set:

|**Test Loss**|**Test Accuracy**|
|----------|---------|
|0.3599|84.40%|




## 📝 License & Notes

- This implementation is for educational/research use.
- Stanford Sentiment Treebank © Stanford NLP
- Google Word2Vec © Google Inc.



