import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块 - 仅最大池化版本"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 仅最大池化
            nn.AvgPool2d(2)   #仅平均池化
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class ShapleyInspiredAttention(nn.Module):
    
    def __init__(self, enc_channels, dec_channels, reduction_ratio=16, use_spatial=True, dropout_rate=0.15):
        super().__init__()
        
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.use_spatial = use_spatial
        self.dropout_rate = dropout_rate
        
        self.drop = nn.Dropout2d(dropout_rate)
        
        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(enc_channels, max(enc_channels // reduction_ratio, 8), 1, bias=False),
            nn.BatchNorm2d(max(enc_channels // reduction_ratio, 8)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate), 
            nn.Conv2d(max(enc_channels // reduction_ratio, 8), enc_channels, 1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        if use_spatial:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(enc_channels + dec_channels, max((enc_channels + dec_channels) // reduction_ratio, 4), 3, padding=1, bias=False),
                nn.BatchNorm2d(max((enc_channels + dec_channels) // reduction_ratio, 4)),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate), 
                nn.Conv2d(max((enc_channels + dec_channels) // reduction_ratio, 4), 1, 3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
        
        # 交叉特征交互
        self.feature_interaction = nn.Sequential(
            nn.Conv2d(enc_channels + dec_channels, enc_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate), 
            nn.Conv2d(enc_channels, enc_channels, 1, bias=False),
            nn.BatchNorm2d(enc_channels)
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.temperature, 1.0)
    
    def forward(self, feat_enc, feat_dec):
        batch_size, _, H, W = feat_enc.shape
        
        # 1. 通道注意力
        channel_weights = self.channel_attention(feat_enc)
        safe_temperature = torch.clamp(self.temperature, min=1e-4, max=1e4)
        channel_weights = torch.sigmoid(channel_weights / safe_temperature)
        feat_enc_channel_weighted = feat_enc * channel_weights
        
        # 2. 空间注意力
        if self.use_spatial:
            if feat_dec.shape[2:] != feat_enc.shape[2:]:
                feat_dec_resized = F.interpolate(feat_dec, size=(H, W), mode='bilinear', align_corners=False)
            else:
                feat_dec_resized = feat_dec
            
            spatial_context = torch.cat([feat_enc_channel_weighted, feat_dec_resized], dim=1)
            spatial_context = self.drop(spatial_context)
            spatial_weights = self.spatial_attention(spatial_context)
            feat_enc_weighted = feat_enc_channel_weighted * spatial_weights
        else:
            feat_enc_weighted = feat_enc_channel_weighted
        
        # 3. 特征交互
        if feat_dec.shape[2:] != feat_enc.shape[2:]:
            feat_dec_resized = F.interpolate(feat_dec, size=(H, W), mode='bilinear', align_corners=False)
        else:
            feat_dec_resized = feat_dec
            
        interacted_features = torch.cat([feat_enc_weighted, feat_dec_resized], dim=1)
        interacted_features = self.drop(interacted_features)
        refined_features = self.feature_interaction(interacted_features)
        
        output = feat_enc_weighted + refined_features
        output = self.drop(output)
        return output


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.attention = ShapleyInspiredAttention(in_channels // 2, in_channels // 2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.drop = nn.Dropout(0.15)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.size() != x2.size():
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
            
        x2 = self.attention(x2, x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.drop(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = DoubleConv(in_channels, in_channels // 4)
        self.drop = nn.Dropout(0.15)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)      # (B,64,H,W)
        x2 = self.down1(x1)   # (B,128,H/2,W/2)
        x3 = self.down2(x2)   # (B,256,H/4,W/4)
        x4 = self.down3(x3)   # (B,512,H/8,W/8)
        x5 = self.down4(x4)   # (B,1024,H/16,W/16)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        seg = self.outc(x)
        return seg


































# 完整代码

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.nn as nn

# class SpatialAwarePoolingSelector(nn.Module):
#     """空间感知的池化选择器"""
#     def __init__(self, in_channels, reduction=16):
#         super(SpatialAwarePoolingSelector, self).__init__()
        
#         self.avg_pool = nn.AvgPool2d(2)
#         self.max_pool = nn.MaxPool2d(2)
        
#         # 更复杂的权重网络，考虑空间信息
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 3, padding=1),
#             nn.BatchNorm2d(in_channels // reduction),  # 添加BN
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.15),  # 添加Dropout
#             nn.Conv2d(in_channels // reduction, 2, 1),  # 输出2个通道的注意力图
#             nn.Softmax(dim=1)  # 每个位置都有avg和max的权重
#         )
        
#         self.drop = nn.Dropout2d(0.15)
        
#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
        
#         # 计算空间注意力权重
#         spatial_weights = self.spatial_attention(x)  # [batch, 2, H, W]
        
#         # 下采样注意力权重以匹配池化后的尺寸
#         spatial_weights = F.avg_pool2d(spatial_weights, 2)
#         w_avg, w_max = spatial_weights[:, 0:1], spatial_weights[:, 1:2]
        
#         # 加权融合
#         output = w_avg * avg_out + w_max * max_out
#         output = self.drop(output)
#         return output

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)
    

# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.pool_conv = nn.Sequential(
#             # nn.MaxPool2d(2),
#             SpatialAwarePoolingSelector(in_channels),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.pool_conv(x)
    

# class ShapleyInspiredAttention(nn.Module):
#     """
#     Shapley-Inspired Attention Module for U-Net Skip Connections
#     基于沙普利值思想的轻量级注意力模块
#     """
    
#     def __init__(self, enc_channels, dec_channels, reduction_ratio=16, use_spatial=True, dropout_rate=0.15):
#         super().__init__()
        
#         self.enc_channels = enc_channels
#         self.dec_channels = dec_channels
#         self.use_spatial = use_spatial
#         self.dropout_rate = dropout_rate
        
#         self.drop = nn.Dropout2d(dropout_rate)
        
#         # 通道注意力分支 - 模拟通道级沙普利值
#         self.channel_attention = nn.Sequential(
#             # 全局上下文编码
#             nn.AdaptiveAvgPool2d(1),
#             # 瓶颈结构减少参数量
#             nn.Conv2d(enc_channels, max(enc_channels // reduction_ratio, 8), 1, bias=False),
#             nn.BatchNorm2d(max(enc_channels // reduction_ratio, 8)),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout_rate), 
#             # 恢复通道数，输出通道重要性权重
#             nn.Conv2d(max(enc_channels // reduction_ratio, 8), enc_channels, 1, bias=False),
#             nn.BatchNorm2d(enc_channels),
#             nn.Sigmoid()  # 输出[0,1]的权重，模拟贡献度
#         )
        
#         # 空间注意力分支 - 模拟空间级沙普利值（可选）
#         if use_spatial:
#             self.spatial_attention = nn.Sequential(
#                 # 同时利用编码器和解码器信息来评估空间重要性
#                 nn.Conv2d(enc_channels + dec_channels, max((enc_channels + dec_channels) // reduction_ratio, 4), 3, padding=1, bias=False),
#                 nn.BatchNorm2d(max((enc_channels + dec_channels) // reduction_ratio, 4)),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout2d(dropout_rate), 
#                 nn.Conv2d(max((enc_channels + dec_channels) // reduction_ratio, 4), 1, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(1),
#                 nn.Sigmoid()
#             )
        
#         # 交叉特征交互 - 模拟联盟价值评估
#         self.feature_interaction = nn.Sequential(
#             nn.Conv2d(enc_channels + dec_channels, enc_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(enc_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout_rate), 
#             nn.Conv2d(enc_channels, enc_channels, 1, bias=False),
#             nn.BatchNorm2d(enc_channels)
#         )
        
#         # 修复：安全地初始化温度参数
#         self.temperature = nn.Parameter(torch.tensor(1.0))
#         # 或者使用register_buffer来避免梯度问题
#         # self.register_buffer('temperature', torch.tensor(1.0))
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         # 特别初始化温度参数
#         nn.init.constant_(self.temperature, 1.0)
    
#     def forward(self, feat_enc, feat_dec):
#         """
#         Args:
#             feat_enc: 编码器特征 [B, C_enc, H, W]
#             feat_dec: 解码器特征 [B, C_dec, H, W]
#         Returns:
#             加权的编码器特征 [B, C_enc, H, W]
#         """

#         batch_size, _, H, W = feat_enc.shape
        
#         # 1. 通道注意力 - 评估每个通道的贡献度
#         channel_weights = self.channel_attention(feat_enc)  # [B, C_enc, 1, 1]
        
#         # 修复：安全地应用温度参数
#         # 方案1：使用clamp确保温度不会太小
#         safe_temperature = torch.clamp(self.temperature, min=1e-4, max=1e4)
#         channel_weights = torch.sigmoid(channel_weights / safe_temperature)
        
#         # 方案2：直接移除温度参数（如果不需要的话）
#         # channel_weights = torch.sigmoid(channel_weights)  # 直接使用sigmoid
        
#         # 方案3：添加小的epsilon防止除零
#         # eps = 1e-8
#         # channel_weights = torch.sigmoid(channel_weights / (self.temperature + eps))
        
#         # 通道级加权
#         feat_enc_channel_weighted = feat_enc * channel_weights
        
#         # 2. 空间注意力 - 评估每个空间位置的贡献度
#         if self.use_spatial:
#             # 调整解码器特征尺寸以匹配编码器特征
#             if feat_dec.shape[2:] != feat_enc.shape[2:]:
#                 feat_dec_resized = F.interpolate(feat_dec, size=(H, W), mode='bilinear', align_corners=False)
#             else:
#                 feat_dec_resized = feat_dec
            
#             # 拼接特征以获取上下文信息
#             spatial_context = torch.cat([feat_enc_channel_weighted, feat_dec_resized], dim=1)
#             spatial_context = self.drop(spatial_context)
#             spatial_weights = self.spatial_attention(spatial_context)  # [B, 1, H, W]
            
#             # 空间级加权
#             feat_enc_weighted = feat_enc_channel_weighted * spatial_weights
#         else:
#             feat_enc_weighted = feat_enc_channel_weighted
        
#         # 3. 特征交互 - 模拟联盟价值函数
#         if feat_dec.shape[2:] != feat_enc.shape[2:]:
#             feat_dec_resized = F.interpolate(feat_dec, size=(H, W), mode='bilinear', align_corners=False)
#         else:
#             feat_dec_resized = feat_dec
            
#         interacted_features = torch.cat([feat_enc_weighted, feat_dec_resized], dim=1)
#         interacted_features = self.drop(interacted_features)
#         refined_features = self.feature_interaction(interacted_features)
        
#         # 残差连接保持信息流
#         output = feat_enc_weighted + refined_features
#         output = self.drop(output)
#         return output
    
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, num_classes=2, bilinear=True):
#         super(Up, self).__init__()
#         self.bilinear = bilinear

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
#         self.attention = ShapleyInspiredAttention(in_channels // 2, in_channels // 2)   # x4 -> up1
#         # print(in_channels // 2)
#         self.conv = DoubleConv(in_channels, out_channels)


#         self.drop = nn.Dropout(0.15)

#     def forward(self, x1, x2):
#         # x1: decoder 上一层输出
#         # x2: encoder 同层特征

#         x1 = self.up(x1)

#         if x1.size() != x2.size():
#             x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
            
#         x2 = self.attention(x2,x1)
            
            
#         x = torch.cat([x1, x2], dim=1)

#         x = self.conv(x)
#         x = self.drop(x)

#         return x

    
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super(OutConv, self).__init__()
#         self.conv1 = DoubleConv(in_channels, in_channels // 4)
        
#         self.drop = nn.Dropout(0.15)

#         self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

#     def forward(self, x):
#         x = self.conv1(x)
        
#         x = self.drop(x)
#         x = self.conv2(x)
#         return x
    

    
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # 编码器
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
       
        
#         # 解码器
#         self.up1 = Up(1024, 512//factor, bilinear)
#         self.up2 = Up(512, 256//factor, bilinear)
#         self.up3 = Up(256, 128//factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)

        
#         self.outc = OutConv(64, 1)     
#     def forward(self, x):
        
#         x1 = self.inc(x)      # (B,64,H,W)
#         x2 = self.down1(x1)   # (B,128,H/2,W/2)
#         x3 = self.down2(x2)   # (B,256,H/4,W/4)
#         x4 = self.down3(x3)   # (B,512,H/8,W/8)
#         x5 = self.down4(x4)                        # (B,1024,H/16,W/16)     
#         x = self.up1(x5, x4)
#         x = self.up2(x,  x3)
#         x = self.up3(x,  x2)
#         x = self.up4(x,  x1)

        
        
#         seg = self.outc(x)


#         return seg


