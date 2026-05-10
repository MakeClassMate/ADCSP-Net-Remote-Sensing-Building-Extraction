# 模型参数
nc = 2 # num_class
bs = 4 # batch_size
ne = 150 # num_epochs
lr = 0.0001 #学习率
num_workers = 4
loss_ = "bce" # (bce)
optimizer_ = "adam" # 优化器

# 模型配置
dataname = "Whu1"#数据集名称
modelname = "OnlyMaxPool" # 选择模型  UNetWithMSAA//UNetWithShapleyAttention
savel_model_path = f"./addsavel_model7/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}"


# # 数据集路径,最后要加反斜杠
data_dir=r'./whu'
train_img_dir = data_dir+r"/train/image/"
train_lab_dir = data_dir+r"/train/label/"
val_img_dir = data_dir+r"/val/image/"
val_lab_dir = data_dir+r"/val/label/"
test_img_dir = data_dir+r"/test/image/"
test_lab_dir = data_dir+r"/test/label/"


# 数据集路径,最后要加反斜杠
# data_dir=r'./whu'
# train_img_dir = data_dir+r"/train_augmented/var_0.05/image/"
# train_lab_dir = data_dir+r"/train_augmented/var_0.05/label/"
# val_img_dir = data_dir+r"/val/image/"
# val_lab_dir = data_dir+r"/val/label/"
# test_img_dir = data_dir+r"/test/image/"
# test_lab_dir = data_dir+r"/test/label/"

# data_dir=r'./macz'
# train_img_dir = data_dir+r"/train/images/"
# train_lab_dir = data_dir+r"/train/labels/"
# val_img_dir = data_dir+r"/val/images/"
# val_lab_dir = data_dir+r"/val/labels/"
# test_img_dir = data_dir+r"/test/images/"
# test_lab_dir = data_dir+r"/test/labels/"