from typing import List
from sklearn.metrics import confusion_matrix


def compute_f1(dataset: str, num_label, gold: List[int], pred: List[int]):
    CM = confusion_matrix(gold, pred)
    if dataset == "HiEve":
        true = sum([CM[i, i] for i in range(2)])
        sum_pred = sum([CM[i, 0:2].sum() for i in range(len(CM))])
        sum_gold = sum([CM[i].sum() for i in range(2)])
        P = true / sum_pred
        R = true / sum_gold
        F1 = 2 * P * R / (P + R)
    return P, R, F1