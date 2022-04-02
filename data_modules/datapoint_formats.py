import networkx as nx
from utils.tools import get_dep_path


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
                triggers = [{'possition': [e1_poss], 'mention': e1['mention'], 'span': (e1['start_char'], e1['end_char'])},
                            {'possition': [e2_poss], 'mention': e2['mention'], 'span': (e2['start_char'], e2['end_char'])}]
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
            triggers = [{'possition': [e1_poss], 'mention': e1['mention'], 'span': e1_span},
                        {'possition': [e2_poss], 'mention': e2['mention'], 'span': e2_span}]
        
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



