import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from getmodel import get_model
from config import *
from metric.evaluator1 import Evaluator
import cv2



import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


model = get_model(modelname)
# 先加载原始 state_dict
state_dict = torch.load(f'{savel_model_path}/best_model.pt', map_location='cuda:0')


filtered_state_dict = {k: v for k, v in state_dict.items() 
                      if not ('total_ops' in k or 'total_params' in k)}

# 加载过滤后的权重（strict=False 允许忽略不匹配的键）
model.load_state_dict(filtered_state_dict, strict=False)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # 将模型移动到选定的设备

# 定义转换函数（如果需要，根据您的模型预处理要求进行调整）
transform = transforms.Compose([  # 将影像调整为模型输入大小
    transforms.ToTensor(),          # 转换为PyTorch张量
])
print(2)

result_folder = f'{savel_model_path}/result0/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    
    
iou_list = []
precision_list = []
recall_list = []
f1_list = []

evaluator = Evaluator(2)
evaluator.reset()
print(3)
image_filenames = os.listdir(test_img_dir)
label_filenames = os.listdir(test_lab_dir)
for i,img_name in enumerate(image_filenames):
    image_path = os.path.join(test_img_dir, img_name)

    label_path = os.path.join(test_lab_dir, label_filenames[i])  # 假设标签与影像文件名相同
    result_path = os.path.join(result_folder, img_name[:-4]+'.jpg')  # 保存预测结果的路径

    # 读取影像和标签
    image1 = cv2.imread(image_path)
    label = cv2.imread(label_path, 0)/255.0
    # print(label)
    print(i)
    

    # 预处理影像
    input_tensor = transform(image1)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # 增加批次维度
    
    label = transform(label)
    
    label = label.float().to(device)

    
    

    # 使用模型进行预测
    with torch.no_grad():
        # output,_,_,_,_ = model(input_tensor)
        output= model(input_tensor)

        label = label.unsqueeze(1).float()

        
        output = torch.sigmoid(output)
        # predicted_label = (output >= 0.5).float().squeeze(0)
        # evaluator.add_batch(pre_image=predicted_label.cpu().numpy(), gt_image=label)
        
        preds = (output >= 0.5).cpu().numpy().astype(np.uint8)
        targets = label.cpu().numpy().astype(np.uint8)
        # 更新评估器
        evaluator.add_batch(preds, targets)
    # 保存预测结果
    predicted_label_np = preds

    predicted_label_np = (predicted_label_np * 255).astype(np.uint8).squeeze()

    cv2.imwrite(result_path, predicted_label_np)
      
    
iou_per_class = evaluator.Intersection_over_Union()
f1_per_class = evaluator.F1()
precision_per_class = evaluator.Precision()
recall_per_class = evaluator.Recall()
oa = evaluator.OA()


print(f"TP: {evaluator.tp}, FP: {evaluator.fp}, TN: {evaluator.tn}, FN: {evaluator.fn}")
print(f"Overall Accuracy (OA): {oa:.6f}")
print(f"Precision per class: [{precision_per_class[0]:.6f}, {precision_per_class[1]:.6f}]")
print(f"Recall per class: [{recall_per_class[0]:.6f}, {recall_per_class[1]:.6f}]")
print(f"F1 per class: [{f1_per_class[0]:.6f}, {f1_per_class[1]:.6f}]")
print(f"Intersection over Union (IoU) per class: [{iou_per_class[0]:.6f}, {iou_per_class[1]:.6f}]")
