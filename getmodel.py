from config import *

from models import lednet3

def get_model(model_name):

    # model_ = lednet.LEDNet(2)
    # model_ = lednet.DHI()
    # model_ = lednet.UNetWithWavelet(in_channels=3, num_classes=2)
    # model_ = lednet.UNet(in_channels=3,out_channels =2)
    # model_ = lednet.UNetCustom(n_channels=3, n_classes=2, bilinear=False)
    # model_ = lednet.UNetWithShapleyAttention(in_channels=3, out_channels=2)
 
    # model_ = lednet.UNetWithMSAA4(in_channels=3, out_channels=2)
    # model_ = lednet.UNetWithMSAA3(in_channels=3, out_channels=2)
    # model_ = lednet.UNet(in_channels=3, out_channels=2)
    # model_ = lednet.UNet(num_classes=2,in_channels=3)
    model_ = lednet3.UNet(n_channels=3, n_classes=1, bilinear=True)
    # model_ = lednet.SegNet(classes=1) #83
    # model_ = lednet.SegNet(input_channels=3, output_channels=1)
    # model_ = lednet.RefineUnet(n_channels=3, n_classes=1)
    # model_ = lednet.MMbNet(inchannel=3, num_class=1)
    # model_ = lednet.MapNet()
    # model_ = lednet.ENet(1)
    # model_ = lednet.AttentionUNet(img_ch=3, output_ch=1)
    # model = lednet4.EdgeSSMUNet(
    #     n_channels=3,
    #     n_classes=1,
    #     bilinear=False,
    #     use_ssm=True,
    #     use_mfse=True,
    #     use_eeam=True,
    #     eeam_mode="full",
    #     detach_reverse_pred=True,
    #     deep_supervision=True,
    # )
    # model_ = lednet4.Segformer(channels=3,num_classes=1)
    # model_ = lednet4.UNetFormer(decode_channels=64,
    #              dropout=0.1,
    #              backbone_name='resnet18.fb_swsl_ig1b_ft_in1k',
    #              pretrained=False,               
    #              window_size=8,
    #              num_classes=1)
    
    # model_ = lednet4.SwinUperNet(num_classes=2)


    return model_

