from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple, Union
from torch import Tensor
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None     # string to use in input/output sentences

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int = None                 # start index in the sentence
    end: int = None                # end index in the sentence
    mention: str = None           # mention of entity
    type: Optional[EntityType] = None   # entity type
    id: Optional[List[int]] = None    # id in the current training/test example

    def to_tuple(self):
        return self.type.natural, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType  # relation type
    head: Entity        # head of the relation
    tail: Entity        # tail of the relation

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class InputExample:
    """
    A single training/ testing example
    """
    dataset: Optional[Dataset] = None
    id: Optional[str] = None
    triggers: List[Entity] = None
    relations: List[Relation] = None
    heads: List[int] = None
    tokens: List[str] = None
    host_sentence_mask: List[int] = None
    dep_path: List[int] = None
    k_walk_nodes: List[int] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data
    Property names are the same names as the corresponding inputs to model.
    """
    input_ids: List[int]
    input_token_ids: List[int]
    input_attention_mask: List[int]
    mapping: Dict[int, List[int]]
    labels: List[int]
    triggers: List[Tuple[List[int], List[int]]]
    # dep_path: List[int]
    adjacent_maxtrix: Tensor
    # scores: List[Tuple[int, int]]                   # distance to triggers.
    # k_walk_nodes: List[int] = None
    host_sentence_mask: List[int] = None
    # input_presentation: Tensor

    
