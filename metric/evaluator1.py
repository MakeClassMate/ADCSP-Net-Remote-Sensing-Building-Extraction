import numpy as np
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, num_class):
        """
        初始化评估器
        
        Args:
            num_class (int): 类别数量，二分类为 2。
        """
        self.num_class = num_class  # 分类数
        self.reset()

    def reset(self):
        """重置评估指标"""
        self.conf_matrix = np.zeros((self.num_class, self.num_class))  # 混淆矩阵
        self.tp = 0  # 真阳性
        self.fp = 0  # 假阳性
        self.fn = 0  # 假阴性
        self.tn = 0  # 真阴性

    def add_batch(self, pre_image: np.ndarray, gt_image: np.ndarray):
        """
        将预测结果和真实标签传递给评估器，用于计算评估指标。
        
        Args:
            pre_image (np.ndarray): 模型的预测标签 (0 或 1)。
            gt_image (np.ndarray): 真实标签 (0 或 1)。
        """
        # 更新混淆矩阵
        cm = confusion_matrix(gt_image.flatten(), pre_image.flatten(), labels=[0, 1])
        self.conf_matrix += cm  # 将批次的混淆矩阵累加

        # 计算 TP, FP, TN, FN
        self.tp += cm[1, 1]  # 真阳性
        self.fp += cm[0, 1]  # 假阳性
        self.fn += cm[1, 0]  # 假阴性
        self.tn += cm[0, 0]  # 真阴性

    def get_tp_fp_tn_fn(self):
        """获取 TP, FP, TN, FN 的值"""
        return self.tp, self.fp, self.tn, self.fn

    def Intersection_over_Union(self):
        """计算每个类别的 IoU"""
        iou_per_class = []
        for i in range(self.num_class):
            intersection = self.conf_matrix[i, i]
            union = self.conf_matrix[i, :].sum() + self.conf_matrix[:, i].sum() - intersection
            iou = intersection / float(union) if union != 0 else float('nan')
            iou_per_class.append(iou)
        return iou_per_class

    def F1(self):
        """计算每个类别的 F1 分数"""
        f1_per_class = []
        precision_per_class = self.Precision()  # 先计算精度
        recall_per_class = self.Recall()  # 再计算召回率
        for i in range(self.num_class):
            p = precision_per_class[i]
            r = recall_per_class[i]
            f1 = 2 * (p * r) / (p + r) if p + r > 0 else float('nan')
            f1_per_class.append(f1)
        return f1_per_class

    def Precision(self):
        """计算每个类别的精度"""
        precision_per_class = []
        for i in range(self.num_class):
            tp = self.conf_matrix[i, i]
            fp = self.conf_matrix[:, i].sum() - tp
            precision = tp / float(tp + fp) if (tp + fp) != 0 else float('nan')
            precision_per_class.append(precision)
        return precision_per_class

    def Recall(self):
        """计算每个类别的召回率"""
        recall_per_class = []
        for i in range(self.num_class):
            tp = self.conf_matrix[i, i]
            fn = self.conf_matrix[i, :].sum() - tp
            recall = tp / float(tp + fn) if (tp + fn) != 0 else float('nan')
            recall_per_class.append(recall)
        return recall_per_class

    def OA(self):
        """计算整体准确率（Overall Accuracy）"""
        total = self.conf_matrix.sum()
        correct = np.trace(self.conf_matrix)  # 真阳性 + 真阴性
        return correct / total if total != 0 else float('nan')