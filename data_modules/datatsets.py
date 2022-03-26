import json
import os
import random
from typing import Dict, List
from data_modules.base_dataset import BaseDataset
from data_modules.input_example import Entity, InputExample, Relation, RelationType


DATASETS: Dict[str, BaseDataset] = {}


def register_dataset(dataset_class: BaseDataset):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(name:str,
                tokenizer: str,
                data_dir: str,
                max_input_length: int,
                seed: int = None,
                split = 'train',):
    '''
    Load a registered dataset
    '''
    return DATASETS[name](
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_input_length=max_input_length,
        seed=seed,
        split=split,
    )


class EEREDataset(BaseDataset):
    relation_types = None
    natural_relation_types = None   # dictionary from relation types given in the dataset to the natural strings to use
    sample = 1

    def load_schema(self):
        self.relation_types = {short: RelationType(short=short, natural=natural)
                            for short, natural in self.natural_relation_types.items()}
    
    def load_data(self, split: str) -> List[InputExample]:
        examples = []
        file_path = os.path.join(self.data_path, f'{split}.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
            for i, datapoint in enumerate(data):
                triggers = [Entity(start=trigger['span'][0], end=trigger['span'][1], mention=trigger['mention'], id=trigger['possition']) 
                            for trigger in datapoint['triggers']]
                
                relations = []
                for relation in datapoint['labels']:
                    relation_type = self.relation_types[relation[2]]
                    if relation_type.short == len(self.natural_relation_types.items()):
                        if random.random() < self.sample:
                            relations.append(Relation(head=triggers[relations[0]], tail=triggers[relations[1]], type=relation_type))
                    else:
                        relations.append(Relation(head=triggers[relations[0]], tail=triggers[relations[1]], type=relation_type))
                
                example = InputExample(
                                    id=i,
                                    triggers=triggers,
                                    relations=relations,
                                    heads=datapoint['heads'],
                                    tokens=datapoint['tokens'],
                                    dep_path=datapoint['dep_path']
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
