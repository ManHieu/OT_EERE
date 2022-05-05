import json
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
from utils.constants import *
from utils.tools import *
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
            end_char = int(line[4]) + len(line[2])
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


# =========================
#       mulerx Reader
# =========================
def mulerx_tsvx_reader(dir_name, file_name):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = '\t'.join(line[1:])
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2])
            my_dict["event_dict"][line[1]] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = line[1]
            event_id2 = line[2]
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
        if sent_id == None:
            print("False to find sent_id")
            print(f'mydict: {my_dict}')
            print(f"event: {event_dict}")
            continue
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        if my_dict["event_dict"][event_id]["mention"] not in ' '.join(my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]):
            print(f'{my_dict["event_dict"][event_id]}  - {my_dict["sentences"][sent_id]}')

    return my_dict


# =========================
#       ESC Reader
# =========================
def cat_xml_reader(dir_name, file_name, intra=True, inter=False):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')

    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None

    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['t_id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['token_span_doc'] = _sent_token_span_doc[k] # from 1
        sent_dict['content'] = ' '.join(v)
        sent_dict['tokens'] = v
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        spacy_token = nlp(sent_dict["content"])
        spacy_tokens = []
        spacy_pos = []
        # for tok in v:
        #     sent_dict['pos'].append(nlp(tok)[0].pos_)
        
        for token in spacy_token:
            spacy_tokens.append(token.text)
            spacy_pos.append(token.pos_)
            if token.dep_ != 'ROOT':
                sent_dict['idx_char_heads'].append(token.head.idx)
                sent_dict['text_heads'].append(token.head.text)
            else:
                sent_dict['idx_char_heads'].append(-1)
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token.dep_)

        sent_dict['d_span'] = (start, start + len(sent_dict['content']))
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        spacy_tokens_span_sent = tokenized_to_origin_span(sent_dict["content"], spacy_tokens)
        mapping = [token_id_lookup(sent_dict["token_span_SENT"], t[0], t[1]) for t in spacy_tokens_span_sent]
        _dep = {}
        _pos = {}
        for char_head, text_head, t_id, pos in zip(sent_dict['idx_char_heads'], sent_dict['text_heads'], mapping, spacy_pos):
            i = 0
            _pos[t_id] = pos
            if char_head != -1:
                for span_sent in sent_dict['token_span_SENT']:
                    if char_head in range(span_sent[0], span_sent[1]):
                        assert text_head in sent_dict['tokens'][i], f"{sent_dict['tokens'][i]} - {text_head} - {span_sent} - {char_head} - {sent_dict}"
                        # sent_dict['heads'].append(i)
                        _dep[t_id] = i
                        break
                    i = i + 1
            else:
                _dep[t_id] = -1

        sent_dict['heads'] = [_dep[id] for id in range(len(sent_dict["token_span_SENT"]))]
        sent_dict['pos'] = [_pos[i] for i in range(len(sent_dict["token_span_SENT"]))]
        assert len(sent_dict['heads']) == len(sent_dict['tokens']), f"{len(sent_dict['heads'])} - {len(sent_dict['tokens'])} - {sent_dict}"
        assert my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]] == sent_dict['content'], f"\n'{sent_dict['content']}' \n '{my_dict['doc_content'][sent_dict['d_span'][0]: sent_dict['d_span'][1]]}'"
        my_dict['sentences'].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(my_dict['doc_id'])
        return None

    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'action' in item.name:
            eid = int(item.attrs['m_id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['t_id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['d_span'] = (start, start + len(mention))
                my_dict['event_dict'][eid]['token_id_list'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = find_sent_id(my_dict['sentences'], mention_span)
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    if intra==True:
        for item in xml_dom.find('relations').children:
            if type(item)== bs4.element.Tag and 'plot_link' in item.name:
                r_id = item.attrs['r_id']
                if item.has_attr('signal'):
                    signal = item.attrs['signal']
                else:
                    signal = ''
                try:
                    r_typ = item.attrs['reltype']
                except:
                    # print(my_dict['doc_id'])
                    # print(item)
                    continue
                cause = item.attrs['causes']
                cause_by = item.attrs['caused_by']
                head = int(item.find('source').attrs['m_id'])
                tail = int(item.find('target').attrs['m_id'])

                assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
                my_dict['relation_dict'][(head, tail)] = r_typ
                
    if inter==True:
        dir_name = './datasets/EventStoryLine/annotated_data/v0.9/'
        inter_dir_name = dir_name.replace('annotated_data', 'evaluation_format/full_corpus') + 'event_mentions_extended/'
        file_name = file_name.replace('.xml.xml', '.xml')
        lines = []
        try:
            with open(inter_dir_name+file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            print("{} is not exit!".format(inter_dir_name+file_name))
        for line in lines:
            rel = line.strip().split('\t')
            r_typ = rel[2]
            head_span, tail_span = get_mention_span(rel[0]), get_mention_span(rel[1])
            # print(head_span, tail_span)
            head, tail = find_m_id(head_span, my_dict['event_dict']), find_m_id(tail_span, my_dict['event_dict'])
            assert head != None and tail != None, f"doc: {inter_dir_name+file_name}, line: {line}, rel: {rel}"

            if r_typ != 'null':
                my_dict['relation_dict'][(head, tail)] = r_typ
    
    return my_dict


if __name__ == '__main__':
    my_dict = tsvx_reader(dir_name="datasets/mulerx/subevent-en-10/test/", file_name="aviation_accidents-week2-nhung-108257_chunk_80.ann.tsvx")
    with open("aviation_accidents-week2-nhung-108257_chunk_80.ann.tsvx", 'w') as f:
        json.dump(my_dict,f, indent=6)
