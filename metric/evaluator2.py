import numpy as np
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def add_batch(self, pre_image, gt_image):
        self.predictions.extend(pre_image.flatten())
        self.ground_truths.extend(gt_image.flatten())

    def compute_metrics(self):
        cm = confusion_matrix(self.ground_truths, self.predictions)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        IoU = tp / (tp + fn + fp)
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'IoU': IoU,
            'confusion_matrix': cm
        }