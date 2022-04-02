from typing import List
from sklearn.metrics import classification_report, confusion_matrix


def compute_f1(dataset: str, num_label: int, gold: List[int], pred: List[int], report: bool=False):
    CM = confusion_matrix(gold, pred)
    if dataset == "HiEve":
        true = sum([CM[i, i] for i in range(2)])
        sum_pred = sum([CM[i, 0:2].sum() for i in range(num_label)])
        sum_gold = sum([CM[i].sum() for i in range(2)])
        P = true / sum_pred
        R = true / sum_gold
        F1 = 2 * P * R / (P + R)
    if report:
        print(f"CM: \n{CM}")
        print("Classification report: \n{}".format(classification_report(gold, pred)))     
        print("  P: {0:.3f}".format(P))
        print("  R: {0:.3f}".format(R))
        print("  F1: {0:.3f}".format(F1))     
    return P, R, F1