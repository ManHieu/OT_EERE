from collections import defaultdict
import pickle
import json
import os
import re
from typing import Dict, List, Optional, Union

from tqdm import tqdm


class ScratchTokenizer(object):
    def __init__(self) -> None:
        self.vocab = None
        self.word_to_id = None
        self.word_counts = None

    def from_file(self, file_path: str):
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                self.vocab, self.word_to_id, self.word_counts = pickle.load(f)
                # print(self.vocab[0])
        else:
            raise "We need pretrained vocab is a pickle file!"
    
    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump((self.vocab, self.word_to_id, self.word_counts), f)

    def fit(self, corpus: Union[List[str], List[List[str]]], tokenized: bool=False):
        word_counts = defaultdict(int)
        if tokenized==True:
            for seq in tqdm(corpus):
                for token in seq:
                    token = re.sub('[^A-Za-z0-9]+', '', token)
                    word_counts[token] += 1
        else:
            for seq in tqdm(corpus):
                tokens = seq.split()
                for token in tokens:
                    token = re.sub('[^A-Za-z0-9]+', '', token)
                    word_counts[token] += 1 
        
        self.word_counts = dict(word_counts)
        self.vocab = {}
        self.word_to_id = {}
        for id, word in enumerate(word_counts.keys(), start=1):
            self.vocab[id] = word
            self.word_to_id[word] = id
        
        self.vocab[0] = '<unk>'
        self.word_to_id['<unk>'] = 0
        # print(self.vocab)
    
    def tokenize(self, seq: Union[List[str], str], tokenized: bool=False):
        if tokenized == False:
            seq_token = seq.split()
        else:
            seq_token = seq
        seq_ids = [self.word_to_id.get(tok) if self.word_to_id.get(tok) != None else 0
                for tok in seq_token]

        return seq_token, seq_ids


if __name__ == '__main__':
    def load_data(data_path, split: str):
        examples = []
        file_path = os.path.join(data_path, f'{split}.json')
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for i, datapoint in enumerate(data):
                # print(f"datapoint: {datapoint}")
                # print(tokens)
                examples.append(datapoint['tokens'])
                
        return examples

    tokenizer = ScratchTokenizer()
    data_name = 'HiEve'
    if data_name == 'HiEve':
        data_path = './datasets/hievents_v2'
    train_set = load_data(data_path, 'train')
    dev_set = load_data(data_path, 'val')
    test_set = load_data(data_path, 'test')
    corpus = train_set + dev_set + test_set
    tokenizer.fit(corpus, tokenized=True)
    tokenizer.save(file_path=data_path + '/tokenizer.pkl')
    # tokenizer.from_file('./datasets/hievents_v2/tokenizer.pkl')



