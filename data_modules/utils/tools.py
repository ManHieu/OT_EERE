from collections import defaultdict
import datetime
import re
from typing import Dict, List, Tuple
import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer, util


nlp = spacy.load("en_core_web_sm")
sim_evaluator = SentenceTransformer('/vinai/hieumdt/all-MiniLM-L12-v1')


# Padding function
def padding(sent, max_sent_len = 194, pad_tok=0):
    one_list = [pad_tok] * max_sent_len # none id 
    one_list[0:len(sent)] = sent
    return one_list


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def tokenized_to_origin_span(text: str, token_list: List[str]):
    token_span = []
    pointer = 0
    for token in token_list:
        start = text.find(token, pointer)
        if start != -1:
            end = start + len(token)
            pointer = end
            token_span.append([start, end])
            assert text[start: end] == token, f"{token}-{text}"
        else:
            token_span.append([-100, -100])
    return token_span


def sent_id_lookup(my_dict, start_char, end_char = None):
    # print(f"my_dict: {my_dict}")
    # print(f"start: {start_char}")
    # print(f"end: {end_char}")
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']


def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        char_ids = range(token_span[0], token_span[1])
        if start_char in char_ids or (end_char-1) in char_ids:
            return index


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC


def id_lookup(span_SENT, start_char, end_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = []
    char_range = set(range(start_char, end_char))
    for i, token_span in enumerate(span_SENT):
        # if token_span[0] <= start_char or token_span[1] >= end_char:
        #     return token_id
        if len(set(range(token_span[0], token_span[1])).intersection(char_range)) > 0:
            token_id.append(i)
    if len(token_id) == 0: 
        raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {} \n end_char: {}".format(span_SENT, start_char, end_char))

    return token_id

def find_common_lowest_ancestor(tree, nodes):
    ancestor = nx.lowest_common_ancestor(tree, nodes[0], nodes[1])
    for node in nodes[2:]:
        ancestor = nx.lowest_common_ancestor(tree, ancestor, node)
    return ancestor


def get_dep_path(tree, nodes):
    # print(tree.edges)
    try:
        ancestor = nx.lowest_common_ancestor(tree, nodes[0], nodes[1])
        for node in nodes[2:]:
            ancestor = nx.lowest_common_ancestor(tree, ancestor, node)

        paths = []
        for node in nodes:
            paths.append(nx.shortest_path(tree, ancestor, node))
        return paths
    except:
        print(tree.edges)
        print(nx.find_cycle(tree, orientation="original"))
        return None


def mapping_subtok_id(subtoks: List[str], tokens: List[str], text: str):
    token_spans = tokenized_to_origin_span(text, tokens)
    subtok_spans = tokenized_to_origin_span(text.lower(), [t.lower() for t in subtoks])

    mapping_dict = defaultdict(list)
    for i, subtok_span in enumerate(subtok_spans, start=1):
        tok_id = token_id_lookup(token_spans, start_char=subtok_span[0], end_char=subtok_span[1])
        if tok_id != None:
            if subtoks[i-1].lower() in tokens[tok_id].lower() or tokens[tok_id].lower() in subtoks[i-1].lower():
                mapping_dict[tok_id].append(i)
            else:
                mapping_dict[tok_id].append(i)
                # print(f"{subtoks[i-1]} - {tokens[tok_id]}") # \n{subtoks} - {subtok_spans}\n{tokens} - {token_spans}
            
    
    # mapping <unk> token:
    for key in range(len(tokens)):
        if mapping_dict.get(key) == None:
            # print(f"haven't_mapping_tok: {tokens[key]}")
            mapping_dict[key] = [random.randint(0, len(tokens)-1)]
    
    # print(f"tokens: {tokens} \nsub_tokens: {subtoks} \nmapping_dict: {mapping_dict}")
    
    return dict(mapping_dict)
    

@torch.no_grad()
def compute_sentences_similar(sent_A: List[str], sent_B: List[str], metric: str='vector_sim'):
    sent_A = ' '.join([word.strip() for word in sent_A])
    sent_B = ' '.join([word.strip() for word in sent_B])

    if metric=='vector_sim':
        embeddings1 = sim_evaluator.encode([sent_A], convert_to_tensor=True)
        embeddings2 = sim_evaluator.encode([sent_B], convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        score = float(cosine_scores[0][0])
    return score


def get_new_poss(poss_in_sent: List[int], new_sid: int, sent_span: Dict[int, Tuple[int, int, int, int]]):
    new_poss = poss_in_sent
    for _new_sid, _, _, sent_len in sent_span.values():
        if _new_sid < new_sid:
            new_poss = [i + sent_len for i in new_poss]
    return new_poss


def find_sent_id(sentences: List[Dict], mention_span: List[int]):
    """
    Find sentence id of mention (ESL)
    """
    for sent in sentences:
        token_span_doc = sent['token_span_doc']
        if set(mention_span) == set(mention_span).intersection(set(token_span_doc)):
            return sent['sent_id']
    
    return None


def get_mention_span(span: str) -> List[int]:
    span = [int(tok.strip()) for tok in span.split('_')]
    return span


def find_m_id(mention: List[int], eventdict:Dict):
    for m_id, ev in eventdict.items():
        # print(mention, ev['mention_span'])
        if mention == ev['mention_span']:
            return m_id
    
    return None

