import json
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
from data_modules.utils.constants import *
from data_modules.utils.tools import *
# from nltk import sent_tokenize
from bs4 import BeautifulSoup as Soup
import csv


# =========================
#       HiEve Reader
# =========================
def tsvx_reader(dir_name, file_name):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = line[1]
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2]) - 1
            my_dict["event_dict"][int(line[1])] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = int(line[1])
            event_id2 = int(line[2])
            rel = line[3]
            my_dict["relation_dict"][(event_id1, event_id2)] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # Split document into sentences
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            if token.dep_ != 'ROOT':
                sent_dict['idx_char_heads'].append(token.head.idx)
                sent_dict['text_heads'].append(token.head.text)
            else:
                sent_dict['idx_char_heads'].append(-1)
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token.dep_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        for char_head, text_head in zip(sent_dict['idx_char_heads'], sent_dict['text_heads']):
            i = 0
            if char_head != -1:
                for span_sent in sent_dict['token_span_SENT']:
                    if char_head == span_sent[0]:
                        assert sent_dict['tokens'][i] == text_head, f"{sent_dict['tokens'][i]} - {text_head} - {span_sent} - {char_head} - {sent_dict}"
                        sent_dict['heads'].append(i)
                        break
                    i = i + 1
            else:
                sent_dict['heads'].append(-1)
        assert len(sent_dict['heads']) == len(sent_dict['tokens'])
        
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        assert my_dict["event_dict"][event_id]["mention"] in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"]], \
            f'{my_dict["event_dict"][event_id]}  - {my_dict["sentences"][sent_id]} - {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"]]}'

    return my_dict


if __name__ == '__main__':
    my_dict = tsvx_reader(dir_name="datasets/hievents_v2/processed/", file_name="article-1126.tsvx")
    with open("article-1126.json", 'w') as f:
        json.dump(my_dict,f, indent=6)
