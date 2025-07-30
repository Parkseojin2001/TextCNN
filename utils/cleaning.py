import re

def convert_score_to_label(score):
    if score <= 0.4:
      return 0
    elif score >= 0.6:
      return 1
    else:
      return None

def replace_str_sst(text):
    """
    Tokenization/string cleaning for the SST dataset
    """
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    return text

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()