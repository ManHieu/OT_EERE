from itertools import combinations
import networkx as nx
from utils.tools import compute_sentences_similar, get_dep_path, get_new_poss
import time 


DATAPOINT = {}


def register_datapoint(func):
    DATAPOINT[str(func.__name__)] = func
    return func


def get_datapoint(type, mydict, doc_info=True):
    return DATAPOINT[type](mydict, doc_info)


@register_datapoint
def hieve_datapoint(my_dict, doc_info=True):
    """
    Format data for HiEve dataset
    """
    data_points = []

    for (eid1, eid2), rel in my_dict['relation_dict'].items():
        e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
        s1, s2 = e1['sent_id'], e2['sent_id']

        if doc_info == True:
            sents_tok_span = []
            tokens = []
            heads = []
            start_tok_id = 0
            for sent in my_dict['sentences']:
                end_tok_id = start_tok_id + len(sent['tokens'])
                sents_tok_span.append((start_tok_id, end_tok_id))
                tokens.extend(sent['tokens'])
                for head in sent['heads']:
                    if head != -1:
                        heads.append(head + start_tok_id)
                    else:
                        heads.append(head)
                assert tokens[start_tok_id: end_tok_id] == sent['tokens'], f"{tokens[start_tok_id: end_tok_id]} - {sent['tokens']} - {(start_tok_id, end_tok_id)}"
                start_tok_id = end_tok_id
            
            e1_poss = e1['token_id'] + sents_tok_span[s1][0]
            e2_poss = e2['token_id'] + sents_tok_span[s2][0]

            if e1['mention'] in tokens[e1_poss]  and e2['mention'] in tokens[e2_poss]:
                triggers = [{'position': [e1_poss], 'mention': e1['mention'], 'span': (e1['start_char'], e1['end_char'])},
                            {'position': [e2_poss], 'mention': e2['mention'], 'span': (e2['start_char'], e2['end_char'])}]
            else:
                print(f"{tokens} - {sents_tok_span}")
                print(f"{e1_poss} - {tokens[e1_poss]} - {e1['mention']}")
                print(f"{e2_poss} - {tokens[e2_poss]} - {e2['mention']}")
                continue

        else:
            heads = []
            tokens = []
            if s1 < s2:
                tokens = tokens + my_dict['sentences'][s1]['tokens'] + my_dict['sentences'][s2]['tokens']
                heads = heads + my_dict['sentences'][s1]['heads']
                for head in my_dict['sentences'][s2]['heads']:
                    if head != -1:
                        heads.append(head + len(my_dict['sentences'][s1]['tokens']))
                    else:
                        heads.append(head)
                
                e1_poss = e1['token_id']
                e2_poss = e2['token_id'] + len(my_dict['sentences'][s1]['tokens'])
                e1_span = (e1['start_char'] - my_dict['sentences'][s1]['sent_start_char'], e1['end_char'] - my_dict['sentences'][s1]['sent_start_char'])
                e2_span = (e2['start_char'] - my_dict['sentences'][s2]['sent_start_char'] + len(my_dict['sentences'][s1]['tokens']), 
                            e2['end_char'] - my_dict['sentences'][s2]['sent_start_char'] + len(my_dict['sentences'][s1]['tokens']))

            elif s2 < s1:
                tokens = tokens + my_dict['sentences'][s2]['tokens'] + my_dict['sentences'][s1]['tokens']
                heads = heads + my_dict['sentences'][s2]['heads']
                for head in my_dict['sentences'][s1]['heads']:
                    if head != -1:
                        heads.append(head + len(my_dict['sentences'][s2]['tokens']))
                    else:
                        heads.append(head)
                
                e2_poss = e2['token_id']
                e1_poss = e1['token_id'] + len(my_dict['sentences'][s2]['tokens'])
                e1_span = (e1['start_char'] - my_dict['sentences'][s1]['sent_start_char'] + len(my_dict['sentences'][s2]['tokens']), 
                           e1['end_char'] - my_dict['sentences'][s1]['sent_start_char'] + len(my_dict['sentences'][s2]['tokens']))
                e2_span = (e2['start_char'] - my_dict['sentences'][s2]['sent_start_char'], e2['end_char'] - my_dict['sentences'][s2]['sent_start_char'])

            else:
                tokens = tokens + my_dict['sentences'][s1]['tokens']
                heads = heads + my_dict['sentences'][s2]['heads']
                e2_poss = e2['token_id']
                e1_poss = e1['token_id']
                e1_span = (e1['start_char'] - my_dict['sentences'][s1]['sent_start_char'], e1['end_char'] - my_dict['sentences'][s1]['sent_start_char'])
                e2_span = (e2['start_char'] - my_dict['sentences'][s2]['sent_start_char'], e2['end_char'] - my_dict['sentences'][s2]['sent_start_char'])
            
            assert e1['mention'] in tokens[e1_poss]
            assert e2['mention'] in tokens[e2_poss] 
            triggers = [{'position': [e1_poss], 'mention': e1['mention'], 'span': e1_span},
                        {'position': [e2_poss], 'mention': e2['mention'], 'span': e2_span}]
        
        dep_tree = nx.DiGraph()
        for head, tail in zip(heads, list(range(len(tokens)))):
            if head != tail:
                dep_tree.add_edge(head, tail)
        dep_paths = get_dep_path(dep_tree, [e1_poss, e2_poss])
        # if dep_paths ==  None:
        #     print(my_dict['sentences'][s1])
        #     print(my_dict['sentences'][s2])
        if dep_paths != None:
            _on_dp = []
            for path in dep_paths:
                _on_dp += path
            _on_dp = set(_on_dp)
            k_walk_nodes = []
            for node in _on_dp:
                k_walk_nodes.extend(list(nx.dfs_tree(dep_tree, node, depth_limit=2).nodes()))
            k_walk_nodes = set(k_walk_nodes)

            on_dp = [0] * (len(tokens) + 1) # ROOT in the last token
            k_walk_nodes_mask  = [0] * (len(tokens) + 1) # ROOT in the last token
            for idx in _on_dp:
                on_dp[idx] = 1
            for idx in k_walk_nodes:
                k_walk_nodes_mask[idx] = 1

            data_point = {
                    'tokens': tokens,
                    'triggers': triggers,
                    'heads': heads,
                    'dep_path': on_dp,
                    'k_walk_nodes': k_walk_nodes_mask,
                    'labels': [(0, 1, rel)]
                }
            data_points.append(data_point)

    return data_points


@register_datapoint
def hieve_datapoint_v2(my_dict, doc_info=True):
    """
    Format data for HiEve dataset which choose the most similar context sentences 
    """
    data_points = []
    simmilar_scores = {}
    for (eid1, eid2), rel in my_dict['relation_dict'].items():
        e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
        s1, s2 = e1['sent_id'], e2['sent_id']
        
        s1_tokens, s2_tokens = my_dict['sentences'][s1]['tokens'], my_dict['sentences'][s2]['tokens']
        scores = []
        for s_id, sent in enumerate(my_dict['sentences']):
            sent_tokens = sent['tokens']
            if s_id != s1 and s_id != s2:
                if simmilar_scores.get((s1, s_id)) == None:
                    s1_sim_score = compute_sentences_similar(s1_tokens, sent_tokens)
                    simmilar_scores[(s1, s_id)] = s1_sim_score
                    simmilar_scores[(s_id, s1)] = s1_sim_score
                else:
                    s1_sim_score = simmilar_scores[(s1, s_id)]
                if simmilar_scores.get((s2, s_id)) == None:
                    s2_sim_score = compute_sentences_similar(s2_tokens, sent_tokens)
                    simmilar_scores[(s2, s_id)] = s2_sim_score
                    simmilar_scores[(s_id, s2)] = s2_sim_score
                else:
                    s2_sim_score = simmilar_scores[(s2, s_id)]
                score = max(s1_sim_score, s2_sim_score)
                scores.append((s_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        # print(f"scores: {scores}")
        top_3_relevances = scores[0: 3] + [(s1, 1), (s2, 1)]

        sents_tok_span = {}
        tokens = []
        host_sentence_mask = []
        heads = []
        start_tok_id = 0
        for i, (sid, _) in enumerate(sorted(top_3_relevances, key=lambda x: x[0])):
            sent = my_dict['sentences'][sid]

            if sid == s1 or sid == s2:
                host_sentence_mask = host_sentence_mask + [1] * len(sent['tokens'])
            else:
                host_sentence_mask = host_sentence_mask + [0] * len(sent['tokens'])

            end_tok_id = start_tok_id + len(sent['tokens'])
            sents_tok_span[sid] = (i, start_tok_id, end_tok_id, len(sent['tokens']))
            
            tokens.extend(sent['tokens'])
            
            for head in sent['heads']:
                if head != -1:
                    heads.append(head + start_tok_id)
                else:
                    heads.append(head)
            assert tokens[start_tok_id: end_tok_id] == sent['tokens'], f"{tokens[start_tok_id: end_tok_id]} - {sent['tokens']} - {(start_tok_id, end_tok_id)}"
            start_tok_id = end_tok_id

        e1_poss = get_new_poss([e1['token_id']], sents_tok_span[s1][0], sents_tok_span)
        e2_poss = get_new_poss([e2['token_id']], sents_tok_span[s2][0], sents_tok_span)

        if e1['mention'] in tokens[e1_poss[0]]  and e2['mention'] in tokens[e2_poss[0]]:
            triggers = [{'position': e1_poss, 'mention': e1['mention']},
                        {'position': e2_poss, 'mention': e2['mention']}]
        else:
            print(f"{tokens} - {sents_tok_span}")
            print(f"{e1_poss} - {tokens[e1_poss]} - {e1['mention']}")
            print(f"{e2_poss} - {tokens[e2_poss]} - {e2['mention']}")
            continue

        dep_tree = nx.DiGraph()
        for head, tail in zip(heads, list(range(len(tokens)))):
            if head != tail:
                dep_tree.add_edge(head, tail)
        dep_paths = get_dep_path(dep_tree, [e1_poss, e2_poss])
        # if dep_paths ==  None:
        #     print(my_dict['sentences'][s1])
        #     print(my_dict['sentences'][s2])
        if dep_paths != None:
            _on_dp = []
            for path in dep_paths:
                _on_dp += path
            _on_dp = set(_on_dp)
            k_walk_nodes = []
            for node in _on_dp:
                k_walk_nodes.extend(list(nx.dfs_tree(dep_tree, node, depth_limit=2).nodes()))
            k_walk_nodes = set(k_walk_nodes)

            on_dp = [0] * (len(tokens) + 1) # ROOT in the last token
            k_walk_nodes_mask  = [0] * (len(tokens) + 1) # ROOT in the last token
            for idx in _on_dp:
                on_dp[idx] = 1
            for idx in k_walk_nodes:
                k_walk_nodes_mask[idx] = 1

            data_point = {
                    'tokens': tokens,
                    'host_sent': host_sentence_mask,
                    'triggers': triggers,
                    'heads': heads,
                    'dep_path': on_dp,
                    'k_walk_nodes': k_walk_nodes_mask,
                    'labels': [(0, 1, rel)]
                }
            data_points.append(data_point)
    return data_points


@register_datapoint
def hieve_datapoint_v3(my_dict, doc_info=True):
    """
    Format data for HiEve dataset which choose the most similar context sentences 
    (two host sentences is a dataopint that mean it can have more than one event pair labeled in a datapoint)
    """
    sentence_pairs = combinations(range(len(my_dict['sentences'])), r=2)
    sentence_pairs = list(sentence_pairs) + [(i, i) for i in range(len(my_dict['sentences']))]
    sentence_pairs = set(sentence_pairs)

    simmilar_scores = {}
    data_points = []
    for s1, s2 in sentence_pairs:
        s1_tokens, s2_tokens = my_dict['sentences'][s1]['tokens'], my_dict['sentences'][s2]['tokens']
        scores = []
        for s_id, sent in enumerate(my_dict['sentences']):
            sent_tokens = sent['tokens']
            if s_id != s1 and s_id != s2:
                if simmilar_scores.get((s1, s_id)) == None:
                    s1_sim_score = compute_sentences_similar(s1_tokens, sent_tokens)
                    simmilar_scores[(s1, s_id)] = s1_sim_score
                    simmilar_scores[(s_id, s1)] = s1_sim_score
                else:
                    s1_sim_score = simmilar_scores[(s1, s_id)]
                if simmilar_scores.get((s2, s_id)) == None:
                    s2_sim_score = compute_sentences_similar(s2_tokens, sent_tokens)
                    simmilar_scores[(s2, s_id)] = s2_sim_score
                    simmilar_scores[(s_id, s2)] = s2_sim_score
                else:
                    s2_sim_score = simmilar_scores[(s2, s_id)]
                score = max(s1_sim_score, s2_sim_score)
                scores.append((s_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        # print(f"scores: {scores}")
        top_3_relevances = scores[0: 3] + [(s1, 1), (s2, 1)]
        top_3_relevances = list(set(top_3_relevances))
        sents_tok_span = {}
        tokens = []
        host_sentence_mask = []
        heads = []
        start_tok_id = 0
        content = ''
        for i, (sid, _) in enumerate(sorted(top_3_relevances, key=lambda x: x[0])):
            sent = my_dict['sentences'][sid]
            content = content + sent['content']

            if sid == s1 or sid == s2:
                host_sentence_mask = host_sentence_mask + [1] * len(sent['tokens'])
            else:
                host_sentence_mask = host_sentence_mask + [0] * len(sent['tokens'])

            end_tok_id = start_tok_id + len(sent['tokens'])
            sents_tok_span[sid] = (i, start_tok_id, end_tok_id, len(sent['tokens']))
            
            tokens.extend(sent['tokens'])
            
            for head in sent['heads']:
                if head != -1:
                    heads.append(head + start_tok_id)
                else:
                    heads.append(head)
            assert tokens[start_tok_id: end_tok_id] == sent['tokens'], f"{tokens[start_tok_id: end_tok_id]} - {sent['tokens']} - {(start_tok_id, end_tok_id)}"
            start_tok_id = end_tok_id

        triggers = []
        labels = []
        for (eid1, eid2), rel in my_dict['relation_dict'].items():
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            _s1, _s2 = e1['sent_id'], e2['sent_id']
            if (_s1 == s1 and _s2 == s2) or (_s1 == s2 and _s2 == s1):
                e1_poss = get_new_poss(e1['token_id'], sents_tok_span[_s1][0], sents_tok_span)
                e2_poss = get_new_poss(e2['token_id'], sents_tok_span[_s2][0], sents_tok_span)
                if e1['mention'] in tokens[e1_poss[0]] and e2['mention'] in tokens[e2_poss[0]]:
                    e1_point = {'position': e1_poss, 'mention': e1['mention'], 'sid': _s1}
                    e2_point = {'position': e2_poss, 'mention': e2['mention'], 'sid': _s2}
                    if e1_point not in triggers:
                        triggers.append(e1_point)
                    if e2_point not in triggers:
                        triggers.append(e2_point)
                    labels.append((triggers.index(e1_point), triggers.index(e2_point), rel))
                else:
                    print(f"{tokens} - {sents_tok_span}")
                    print(f"{e1_poss} - {tokens[e1_poss]} - {e1['mention']}")
                    print(f"{e2_poss} - {tokens[e2_poss]} - {e2['mention']}")
                    continue
                
        if len(labels) > 0:
            data_point = {
                        'tokens': tokens,
                        'host_sent': host_sentence_mask,
                        'content': content,
                        'triggers': triggers,
                        'heads': heads,
                        'labels': labels
                    }
            data_points.append(data_point)
        
    # num_labels = sum([len(datapoint['labels']) for datapoint in data_points])
    # assert len(my_dict['relation_dict'].items()) == num_labels, print(f"{len(my_dict['relation_dict'].items())} - {num_labels}")
    
    return data_points


@register_datapoint
def mulerx_datapoint(my_dict, doc_info=True):
    """
    Format data for Mulerx dataset which choose the most similar context sentences 
    (two host sentences is a dataopint that mean it can have more than one event pair labeled in a datapoint)
    """
    sentence_pairs = combinations(range(len(my_dict['sentences'])), r=2)
    sentence_pairs = list(sentence_pairs) + [(i, i) for i in range(len(my_dict['sentences']))]
    sentence_pairs = set(sentence_pairs)

    simmilar_scores = {}
    data_points = []
    for s1, s2 in sentence_pairs:
        # print(f"scores: {scores}")
        top_3_relevances = [(sid, 1) for sid in range(len(my_dict['sentences']))]
        top_3_relevances = list(set(top_3_relevances))
        sents_tok_span = {}
        tokens = []
        host_sentence_mask = []
        heads = []
        start_tok_id = 0
        content = ''
        for i, (sid, _) in enumerate(sorted(top_3_relevances, key=lambda x: x[0])):
            sent = my_dict['sentences'][sid]
            content = content + sent['content']

            if sid == s1 or sid == s2:
                host_sentence_mask = host_sentence_mask + [1] * len(sent['tokens'])
            else:
                host_sentence_mask = host_sentence_mask + [0] * len(sent['tokens'])

            end_tok_id = start_tok_id + len(sent['tokens'])
            sents_tok_span[sid] = (i, start_tok_id, end_tok_id, len(sent['tokens']))
            
            tokens.extend(sent['tokens'])
            
            for head in sent['heads']:
                if head != -1:
                    heads.append(head + start_tok_id)
                else:
                    heads.append(head)
            assert tokens[start_tok_id: end_tok_id] == sent['tokens'], f"{tokens[start_tok_id: end_tok_id]} - {sent['tokens']} - {(start_tok_id, end_tok_id)}"
            start_tok_id = end_tok_id

        triggers = []
        labels = []
        for (eid1, eid2), rel in my_dict['relation_dict'].items():
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            if e1.get('token_id') == None or e2.get('token_id') == None:
                continue
            _s1, _s2 = e1['sent_id'], e2['sent_id']
            if (_s1 == s1 and _s2 == s2) or (_s1 == s2 and _s2 == s1):
                e1_poss = get_new_poss(e1['token_id'], sents_tok_span[_s1][0], sents_tok_span)
                e2_poss = get_new_poss(e2['token_id'], sents_tok_span[_s2][0], sents_tok_span)
                e1_point = {'position': e1_poss, 'mention': e1['mention'], 'sid': _s1}
                e2_point = {'position': e2_poss, 'mention': e2['mention'], 'sid': _s2}
                if e1_point not in triggers:
                    triggers.append(e1_point)
                if e2_point not in triggers:
                    triggers.append(e2_point)
                labels.append((triggers.index(e1_point), triggers.index(e2_point), rel))
                if  any([tokens[i] not in e1['mention'] for i in e1_poss]) or any([tokens[i] not in e2['mention'] for i in e2_poss]):
                    print(f"{tokens} - {sents_tok_span}")
                    print(f"{e1_poss} - {' '.join([tokens[i] for i in e1_poss])} - {e1['mention']}")
                    print(f"{e2_poss} - {' '.join([tokens[i] for i in e2_poss])} - {e2['mention']}")
                    continue
                
        if len(labels) > 0:
            data_point = {
                        'tokens': tokens,
                        'content': content,
                        'host_sent': host_sentence_mask,
                        'triggers': triggers,
                        'heads': heads,
                        'labels': labels
                    }
            data_points.append(data_point)
        
    # num_labels = sum([len(datapoint['labels']) for datapoint in data_points])
    # assert len(my_dict['relation_dict'].items()) == num_labels, print(f"{len(my_dict['relation_dict'].items())} - {num_labels}")
    
    return data_points


@register_datapoint
def ESL_datapoint(my_dict, intra=True, inter=False):
    """
    Format data for EventStoryLine corpus 
    """
    data_points = []
    if intra == True and inter==False:
        sentence_pairs = [(i, i) for i in range(len(my_dict['sentences']))]
    else:
        sentence_pairs = combinations(range(len(my_dict['sentences'])), r=2)
        sentence_pairs = list(sentence_pairs) + [(i, i) for i in range(len(my_dict['sentences']))]
        sentence_pairs = set(sentence_pairs)
    
    simmilar_scores = {}
    for s1, s2 in sentence_pairs:
        s1_tokens, s2_tokens = my_dict['sentences'][s1]['tokens'], my_dict['sentences'][s2]['tokens']
        scores = []
        time_start = time.time()
        for s_id, sent in enumerate(my_dict['sentences']):
            sent_tokens = sent['tokens']
            if s_id != s1 and s_id != s2:
                if simmilar_scores.get((s1, s_id)) == None:
                    s1_sim_score = compute_sentences_similar(s1_tokens, sent_tokens)
                    simmilar_scores[(s1, s_id)] = s1_sim_score
                    simmilar_scores[(s_id, s1)] = s1_sim_score
                else:
                    s1_sim_score = simmilar_scores[(s1, s_id)]
                if simmilar_scores.get((s2, s_id)) == None:
                    s2_sim_score = compute_sentences_similar(s2_tokens, sent_tokens)
                    simmilar_scores[(s2, s_id)] = s2_sim_score
                    simmilar_scores[(s_id, s2)] = s2_sim_score
                else:
                    s2_sim_score = simmilar_scores[(s2, s_id)]
                score = max(s1_sim_score, s2_sim_score)
                scores.append((s_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        # print(f"scores: {scores}")
        top_3_relevances = scores[0: 3] + [(s1, 1), (s2, 1)]
        top_3_relevances = list(set(top_3_relevances))
        sents_tok_span = {}
        tokens = []
        host_sentence_mask = []
        heads = []
        start_tok_id = 0
        content = ''
        for i, (sid, _) in enumerate(sorted(top_3_relevances, key=lambda x: x[0])):
            sent = my_dict['sentences'][sid]
            content = content + sent['content']

            if sid == s1 or sid == s2:
                host_sentence_mask = host_sentence_mask + [1] * len(sent['tokens'])
            else:
                host_sentence_mask = host_sentence_mask + [0] * len(sent['tokens'])

            end_tok_id = start_tok_id + len(sent['tokens'])
            sents_tok_span[sid] = (i, start_tok_id, end_tok_id, len(sent['tokens']))
            
            tokens.extend(sent['tokens'])
            
            for head in sent['heads']:
                if head != -1:
                    heads.append(head + start_tok_id)
                else:
                    heads.append(head)
            assert tokens[start_tok_id: end_tok_id] == sent['tokens'], f"{tokens[start_tok_id: end_tok_id]} - {sent['tokens']} - {(start_tok_id, end_tok_id)}"
            start_tok_id = end_tok_id
        # print(f"time: {time.time() - time_start}")
        triggers = []
        for eid, event in my_dict['event_dict'].items():
            if event['sent_id'] in [s1, s2]:
                new_pos = get_new_poss(event['token_id_list'], sents_tok_span[event['sent_id']][0], sents_tok_span)
                e_point = {
                    'eid': eid,
                    'position': list(range(new_pos[0], new_pos[-1] + 1)),
                    'mention': event['mention'], 
                    'sid': event['sent_id']
                }
                if e_point['mention'] != ' '.join([tokens[i] for i in e_point['position']]):
                    print(f"e_point: {e_point} - {' '.join([tokens[i] for i in e_point['position']])}")
                    print(f"tokens: {tokens}")
                triggers.append(e_point)
        
        labels = []
        event_pairs = combinations(triggers, 2)
        for e1, e2 in event_pairs:
            eid1, eid2 = e1['eid'], e2['eid']
            rel = my_dict['relation_dict'].get((eid1, eid2))
            _rel = my_dict['relation_dict'].get((eid2, eid1))
            if rel != None:
                relation = (triggers.index(e1), triggers.index(e2), rel)
            elif _rel != None:
                relation = (triggers.index(e2), triggers.index(e1), _rel)
            else:
                relation = (triggers.index(e1), triggers.index(e2), 'NoRel')
            
            if relation[-1] == 'FALLING_ACTION':
                relation = (relation[1], relation[0], 'PRECONDITION')
            labels.append(relation)
        
        if len(labels) > 0:
            data_point = {
                        'tokens': tokens,
                        'content': content,
                        'host_sent': host_sentence_mask,
                        'triggers': triggers,
                        'heads': heads,
                        'labels': labels
                    }
            data_points.append(data_point)

    return data_points