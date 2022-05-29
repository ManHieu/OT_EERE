import json
import os
import random
from typing import Dict, List, Tuple
from data_modules.base_dataset import BaseDataset
from data_modules.input_example import Entity, InputExample, Relation, RelationType


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
        file_path = os.path.join(self.data_path, f'{split}.json')
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} for split {split} of {self.name} with the sample rate is {self.sample}")
            for i, datapoint in enumerate(data):
                # print(f"datapoint: {datapoint}")
                triggers = [Entity(mention=trigger['mention'], id=trigger['position']) 
                            for trigger in datapoint['triggers']]
                
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
                                        # dep_path=datapoint['dep_path'],
                                        # k_walk_nodes=datapoint['k_walk_nodes'],
                                        host_sentence_mask=datapoint['host_sent']
                    )
                    examples.append(example)
                
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
