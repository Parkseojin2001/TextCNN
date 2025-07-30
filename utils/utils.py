import os
import pandas as pd
from collections import Counter
from utils.cleaning import clean_str_sst, replace_str_sst, convert_score_to_label
from nltk.tokenize import TreebankWordTokenizer

folder_dir = "./data/"

def load_sst_data(file_name, delimiter=None, skip_header=True):
    file_path = os.path.join(folder_dir, file_name)

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    if skip_header:
        lines = lines[1:]

    def try_fix(text):
        try:
            return text.encode('latin1').decode('utf-8')
        except:
            return text

    fixed_lines = [try_fix(line.strip()) for line in lines]

    if delimiter:
        return [line.split(delimiter) for line in fixed_lines]
    else:
        return fixed_lines

def load_sentiment_labels(file_name):
  file_path = os.path.join(folder_dir, file_name)
  return pd.read_csv(file_path, sep='|')

def sentence_labeling(sentences, phrase, labels):
  data = []

  for idx, sentence in sentences:
    if sentence not in phrase.keys():
      print(idx, sentence)
      continue

    pid = phrase[sentence]
    score = labels[pid]
    label = convert_score_to_label(score)

    if label is not None:
      data.append((sentence, label))

  return data

def split_data(sentences, split_idx):
  return [(idx, replace_str_sst(sentence)) for idx, sentence in sentences if idx in split_idx]

def build_vocab(all_tokens):
  counter = Counter(all_tokens)
  vocab = {"<PAD>": 0, "<UNK>": 1}
  for idx, (token, _) in enumerate(counter.most_common(), start=2):
    vocab[token] = idx
  return vocab

def encode(tokens, vocab, max_len):
  token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]

  if len(token_ids) < max_len:
    token_ids += [vocab['<PAD>']] * (max_len - len(token_ids))

  else:
    token_ids = token_ids[:max_len]

  return token_ids

tokenizer = TreebankWordTokenizer()

def load_and_prepare_data():
  all_sentences = load_sst_data('datasetSentences.txt', delimiter ='\t')
  split_idx = load_sst_data('datasetSplit.txt', delimiter=',')
  phrase_dict = {phrase[0]: int(phrase[1]) for phrase in load_sst_data('dictionary.txt', delimiter='|', skip_header=False)}

  sentiment_df = load_sentiment_labels('sentiment_labels.txt')
  sentiment_labels = dict(zip(sentiment_df['phrase ids'], sentiment_df['sentiment values']))

  train_idx = [split[0] for idx, split in enumerate(split_idx) if split[1].strip() == '1']
  test_idx = [split[0] for idx, split in enumerate(split_idx) if split[1].strip() == '2']
  valid_idx = [split[0] for idx, split in enumerate(split_idx) if split[1].strip() == '3']
  
  train_sent = split_data(all_sentences, train_idx)
  valid_sent = split_data(all_sentences, valid_idx)
  test_sent  = split_data(all_sentences, test_idx)
  
  train_labeled = sentence_labeling(train_sent, phrase_dict, sentiment_labels)
  valid_labeled = sentence_labeling(valid_sent, phrase_dict, sentiment_labels)
  test_labeled  = sentence_labeling(test_sent, phrase_dict, sentiment_labels)
  
  # Clean & tokenize
  train_cleaned = [(clean_str_sst(sent), label) for sent, label in train_labeled]
  valid_cleaned = [(clean_str_sst(sent), label) for sent, label in valid_labeled]
  test_cleaned  = [(clean_str_sst(sent), label) for sent, label in test_labeled]

  train_tokens = [(tokenizer.tokenize(sent), label) for sent, label in train_cleaned]
  valid_tokens = [(tokenizer.tokenize(sent), label) for sent, label in valid_cleaned]
  test_tokens  = [(tokenizer.tokenize(sent), label) for sent, label in test_cleaned]
  
  # all tokens for vocab
  all_tokens = [tok for sent, _ in train_tokens for tok in sent]

  return train_tokens, valid_tokens, test_tokens, all_tokens