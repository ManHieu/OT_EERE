from collections import defaultdict
import json
import re
from typing import Dict, List, Optional, Union


class ScratchTokenizer(object):
    def __init__(self) -> None:
        self.vocab = None
        self.word_counts = None

    def from_file(self, file_path: str):
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        else:
            raise "We need pretrained vocab is a .json file!"
    
    def save_vocab(self, file_path: str):
        with open(file_path, 'w', encoding='UTF-8') as f:
            json.dump(self.vocab, f, indent=6)

    def fit(self, corpus: Union[List[str], List[List[str]]], tokenized: bool=False):
        word_counts = defaultdict(int)
        if tokenized==True:
            for seq in corpus:
                for token in seq:
                    token = re.sub('[^A-Za-z0-9]+', '', token)
                    word_counts[token] += 1
        else:
            for seq in corpus:
                tokens = seq.split()
                for token in tokens:
                    token = re.sub('[^A-Za-z0-9]+', '', token)
                    word_counts[token] += 1 
        

