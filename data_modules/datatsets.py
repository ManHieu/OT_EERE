import json
import os
import random
from typing import Dict, List, Tuple
import pickle
from tqdm import tqdm
from data_modules.base_dataset import BaseDataset
from data_modules.input_example import Entity, InputExample, Relation, RelationType
from networkx import nx
from data_modules.utils.tools import get_dep_path


DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(name:str,
                tokenizer: str,
                encoder: str,
                data_dir: str,
                max_input_length: int,
                seed: int = None,
                split = 'train',
                range_dist = None):
    '''
    Load a registered dataset
    '''
    return DATASETS[name](
        tokenizer=tokenizer,
        encoder_model=encoder,
        data_dir=data_dir,
        max_input_length=max_input_length,
        seed=seed,
        split=split,
        range_dist=range_dist
    )


class EEREDataset(BaseDataset):
    relation_types = None
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use
    sample = 1

    def load_schema(self):
        self.relation_types = {natural: RelationType(short=short, natural=natural)
                            for short, natural in self.natural_relation_types.items()}
    
    def load_data(self, split: str, range_dist: Tuple[int, int]=None) -> List[InputExample]:
        examples = []
        self.load_schema()
        cache = os.path.join(self.data_path, f'cache_{split}.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                examples = pickle.load(f)
            return examples

        file_path = os.path.join(self.data_path, f'{split}.json')
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} for split {split} of {self.name} with the sample rate is {self.sample}")
            for i, datapoint in tqdm(enumerate(data)):
                # print(f"datapoint: {datapoint}")
                dep_tree = nx.DiGraph()
                for head, tail in zip(datapoint['heads'], list(range(len(datapoint['tokens'])))):
                    if head != tail:
                        dep_tree.add_edge(head, tail)
                triggers = [Entity(mention=trigger['mention'], id=trigger['position']) 
                            for trigger in datapoint['triggers']]

                dep_paths = get_dep_path(dep_tree, [t.id[0] for t in triggers])
                # if dep_paths ==  None:
                #     print(my_dict['sentences'][s1])
                #     print(my_dict['sentences'][s2])
                if dep_paths != None:
                    _on_dp = []
                    for path in dep_paths:
                        _on_dp += path
                    # _on_dp = set(_on_dp)
                on_dp = [0] * (len(datapoint['tokens']) + 1) # ROOT in the last token
                for idx in _on_dp:
                    on_dp[idx] = 1
                relations = []
                for relation in datapoint['labels']:

                    dist = abs(triggers[relation[0]].id[0] - triggers[relation[1]].id[0])
                    if range_dist != None:
                        if dist > range_dist[1] or dist < range_dist[0]:
                            continue

                    relation_type = self.relation_types[relation[2]]
                    if relation_type.natural == 'NoRel':
                        if random.uniform(0, 1) < self.sample:
                            relations.append(Relation(head=triggers[relation[0]], tail=triggers[relation[1]], type=relation_type))
                    else:
                        relations.append(Relation(head=triggers[relation[0]], tail=triggers[relation[1]], type=relation_type))
                if len(relations) >= 1:
                    example = InputExample(
                                        id=i,
                                        content=datapoint['content'],
                                        triggers=triggers,
                                        relations=relations,
                                        heads=datapoint['heads'],
                                        tokens=datapoint['tokens'],
                                        dep_path=on_dp,
                                        # k_walk_nodes=datapoint['k_walk_nodes'],
                                        host_sentence_mask=datapoint['host_sent']
                    )
                    examples.append(example)
        with open(cache, 'wb') as f:
            pickle.dump(examples, f, pickle.HIGHEST_PROTOCOL)
        return examples


@register_dataset
class HiEveDataset(EEREDataset):
    name = 'HiEve'
    sample = 0.4

    natural_relation_types = {
                            0: "SuperSub", 
                            1: "SubSuper", 
                            2: "Coref", 
                            3: "NoRel"
                            }


@register_dataset
class SubEventMulerxDataset(EEREDataset):
    name = 'subevent_mulerx'
    sample = 1.0

    natural_relation_types = {
                            0: "SuperSub", 
                            1: "SubSuper",  
                            2: "NoRel"
                            }


@register_dataset
class ESLDataset(EEREDataset):
    name = 'ESL'
    sample = 1.0

    natural_relation_types = {
                            0: "PRECONDITION", 
                            1: "NoRel"
                            }
