import numpy as np
np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)


CUDA = torch.cuda.is_available()

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
POS_DICT = {"None": 0, "ADJ": 1, "ADP": 2, "ADV": 3, "AUX": 4, "CONJ": 5, "CCONJ": 6, "DET": 7, 
            "INTJ": 8, "NOUN": 9, "NUM": 10, "PART": 11, "PRON": 12, "PROPN": 13, "PUNCT": 14, 
            "SCONJ": 15, "SYM": 16, "VERB": 17, "X": 18, "SPACE": 19, "UNK":20}

HIEVE_LABEL_DICT = {"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
HIEVE_NUM_DICT = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}

