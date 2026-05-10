import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_dataloader
from config import *
from getmodel import get_model
import torch.nn.functional as F
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from metric.evaluator import Evaluator
import numpy as np
from PIL import Image
from thop import profile
import datetime
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


def Combined_Bce_Dice_Loss(y_pred, y_true , bce_fn, dice_fn):
    # 6
    bce = bce_fn(y_pred, y_true)
    y_pred = torch.sigmoid(y_pred)
    dice = dice_fn(y_pred, y_true)
    return 0.7 * bce + 0.2 * dice



def train_model(model, train_loader, val_loader, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(savel_model_path):
        os.makedirs(savel_model_path)
        
    save_subdir = os.path.join(savel_model_path, 'models')
    os.makedirs(save_subdir, exist_ok=True)
    seed_everything(43)

    model.to(device)
    

    
    
    # criterion1 = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失
    # criterion = nn.BCELoss()
    bce_fn= nn.BCEWithLogitsLoss()
    dice_fn= SoftDiceLoss()
    bce_fn.to(device)
    dice_fn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    


    train_loss_history = []
    val_loss_history = []
    iou_history = []
    f1_history = []
    precision_history = []
    recall_history = []

    
    
    # 初始化最佳验证损失
    best_iou = -float('inf')  # 初始化为负无穷大
    best_f1 = -float('inf')  # 初始化为负无穷大

    # 创建保存模型的目录
    os.makedirs(savel_model_path, exist_ok=True)
    # 初始化 Evaluator
    num_classes = 2  # 二分类问题
    evaluator = Evaluator(num_classes)

    for epoch in range(num_epochs):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)  # 重置统计
        torch.cuda.empty_cache()  # 可选：清理缓存碎片，使测量更干净
        train_loss = 0.0
        a=0.8
        b=0.2
        c=0.1
        step = 0
        for images, labels in train_loader:
            step+=1
            images, labels = images.to(device),labels.to(device)
            output = model(images)

            
            
            labels = labels.unsqueeze(1).float()
            
            loss = criterion(output,labels)

            
            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            

            train_loss += loss.item()

            print("Epoch[%d] step[%d/%d]->loss:%.4f" %
                  (epoch+1,step, len(train_loader), loss.item()))
            # print("loss1",loss1)
            # print("loss2",loss2)
            # print("loss3",loss3)
        
         # 获取当前epoch的峰值显存
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Epoch {epoch+1} Peak GPU Memory: {peak_memory_mib:.2f} MiB")

        train_loss = train_loss / len(train_loader)
        train_loss_history.append(train_loss)
        

        # 在验证集上进行评估
        model.eval()
        val_loss = 0.0
        evaluator.reset()  # 重置评估器
        # print_memory_usage(f"Epoch {epoch+1} - Start Eval")

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) 

                output = model(images) 

    
                labels = labels.unsqueeze(1).float()
                loss = criterion(output,labels)
                val_loss += loss.item()
                # 将输出转换为二值标签
                output = torch.sigmoid(output)
                preds = (output >= 0.5).cpu().numpy().astype(np.uint8)
                targets = labels.cpu().numpy().astype(np.uint8)
                # 更新评估器
                evaluator.add_batch(preds, targets)
        val_loss = val_loss / len(val_loader)

        val_loss_history.append(val_loss)
        # 计算评估指标
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        precision_per_class = evaluator.Precision()
        recall_per_class = evaluator.Recall()
        
        # 保存当前 epoch 的指标
        iou_history.append(iou_per_class[1])  # 通常我们关注正类
        f1_history.append(f1_per_class[1])
        precision_history.append(precision_per_class[1])
        recall_history.append(recall_per_class[1])
        
        # 打印评估指标
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"IoU: {iou_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
        print(f"Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}")
        
        # 更新学习率
        scheduler.step()
        

        
        # 根据 F1 分数保存最佳模型
        if iou_per_class[1] > best_iou:
            best_iou = iou_per_class[1]
            best_f1 = f1_per_class[1]
            torch.save(model.state_dict(), f"{savel_model_path}/best_model.pt")
            
        print(f"Best model saved with iou score: {best_iou:.4f}")
        print(f"Best model saved with f1 score: {best_f1:.4f}")

        # torch.save(model.state_dict(), f"{savel_model_path}/last_model.pt")
    model_save_path = os.path.join(save_subdir, f'model_epoch_{epoch+1}.pt')
    torch.save(model.state_dict(), model_save_path)

    # 绘制损失变化图
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_loss_history, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{savel_model_path}/loss.jpg')
    plt.show()

    # 绘制 IoU, F1, Precision, Recall 曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), iou_history, label="IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), f1_history, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), precision_history, label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), recall_history, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{savel_model_path}/metrics.jpg')
    plt.show()
    
if __name__ == "__main__":

    # 准备数据加载器
    train_loader = prepare_dataloader(train_img_dir, train_lab_dir, batch_size=bs, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = prepare_dataloader(val_img_dir, val_lab_dir, batch_size=2, shuffle=False, drop_last=False, num_workers=num_workers)
    model = get_model(modelname)
    train_model(model, train_loader, val_loader, ne, lr)