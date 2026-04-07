import math
import numpy as np
# PyTorch
import torch
from torch import NoneType, cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
from unet import UNet
from modified_unet import M_UNet 
from keras_unet import K_UNet 
import segmentation_models_pytorch as smp 
from SelfONN import SuperONN1d, SuperONN2d, SelfONN1d, SelfONN2d
from SelfONN_Pretrained_UNet import SelfONNUnet
from selfonn_unet_selflayer import SelfONNUnet_new
import deeplabv3_selfonn_model
import SelfONN_decoders
import Unet3Plus_SMP
import SMP_CSCAUNet
from SA_UNet import SA_UNet
from ACCUnet import ACC_UNet
from CSC_Unet import CSC_UNet
from CSCUNetPP import CSCUNetPP
from CSCUNetPPP import CSCUNet3PlusDeepSup
from CSCAUNet import CSCAUNet
from densnet121_CSCAUNet import CSCAUNet_DenseNet
from SelfONN_densenet121_CSCAUNet import SelfONN_CSCAUNet_DenseNet
from FullSelfONN_densenet121_CSCAUNet import FullSelfONN_CSCAUNet_DenseNet
import SuperONN_decoders
import dual_enc_unet

def reset_function_generic(m):
    if hasattr(m,'reset_parameters'): 
        m.reset_parameters()

def get_pretrained_model(parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift, U_init_features,in_channels,out_channels,train_on_gpu,multi_gpu, decoder_attention = None, activation = None, unfolding = 1):
    dec_dict = {
        'Unet': smp.Unet,
        'UnetPlusPlus' : smp.UnetPlusPlus,
        'MAnet' : smp.MAnet,
        'Linknet' : smp.Linknet,
        'FPN' : smp.FPN,
        'PSPNet' : smp.PSPNet,
        'PAN' : smp.PAN,
        'DeepLabV3' : smp.DeepLabV3,
        'DeepLabV3Plus' : smp.DeepLabV3Plus,
        'UPerNet' : smp.UPerNet,
        'Segformer' : smp.Segformer,
        'DPT': smp.DPT,
        'Unet3Plus' : Unet3Plus_SMP.Unet3Plus,
        'DualEncoderUnet' : dual_enc_unet.DualEncoderUnet,
        'SelfONN_Unet' : SelfONN_decoders.SelfONNUnet,
        'SelfONN_UnetPlusPlus' : SelfONN_decoders.SelfONNUnetPlusPlus,
        'SelfONN_ResUnet' : SelfONN_decoders.SelfONN_ResUnet,
        'SelfONN_FPN' : SelfONN_decoders.SelfONN_FPN,
        'SelfONN_DeepLabV3' : deeplabv3_selfonn_model.SelfONN_DeepLabV3,
        'SelfONN_DeepLabV3Plus' : deeplabv3_selfonn_model.SelfONN_DeepLabV3Plus,
        'SelfONN_MAnet': SelfONN_decoders.SelfONN_MAnet,
        'SelfONN_CSCAUnet' : SelfONN_decoders.SelfONN_CSCAUnet,
        'CSCAUnet' : SMP_CSCAUNet.CSCAUnet,
        'SuperONN_Unet' : SuperONN_decoders.SuperONNUnet,
    }


    #loading models from the SMP library
    print('Initialized with', model_type,'\n')
    if model_type == "SMP":
        if len(model_to_load.split('*')) == 3:
            enc_name_1, enc_name_2, dec_name = model_to_load.split('*')
            dec_name = 'DualEncoder' + dec_name
            print(dec_name)
            model = dec_dict[dec_name](enc_name_1, enc_name_2, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation)
        else:
            enc_name, dec_name = model_to_load.split('*')
            if dec_name == 'Unet' or dec_name == 'UnetPlusPlus' or dec_name == 'Unet3Plus':
                print('decoder_attention:', str(decoder_attention), '\n')
                model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, decoder_attention_type = decoder_attention, in_channels=in_channels, classes=out_channels, activation=activation)
            else:
                if dec_name == 'PAN':
                    model = dec_dict[dec_name](enc_name, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation)
                elif dec_name == 'CSCAUnet':
                    model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation, decay=unfolding)
                else:
                    model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation)

    #loading models from the ONN Decoders file
    elif model_type == "ONN_dec":
        enc_name, dec_name = model_to_load.split('*')
        if 'Self' in dec_name:
            if dec_name == 'SelfONN_Unet' or dec_name == 'SelfONN_UnetPlusPlus' or dec_name == 'SelfONN_ResUnet':
                print('decoder_attention:', str(decoder_attention), '\n')
                model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, decoder_attention_type = decoder_attention, in_channels=in_channels, classes=out_channels,  q_order = q_order, max_shift = max_shift, activation=activation)
            elif dec_name == 'SelfONN_CSCAUnet':
                model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation, decay=unfolding, q = q_order)
            else:
                model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels,  q_order = q_order, max_shift = max_shift, activation=activation)
        else:
            if dec_name == 'SuperONN_Unet' or dec_name == 'SuperONN_UnetPlusPlus' or dec_name == 'SuperONN_ResUnet':
                print('decoder_attention:', str(decoder_attention), '\n')
                model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, decoder_attention_type = decoder_attention, in_channels=in_channels, classes=out_channels,  q_order = q_order, max_factor = max_shift, activation=activation)
            
    #loading custom built models        
    else:
        if model_to_load == 'UNet':  
            model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
        elif model_to_load == 'M_UNet': 
            model = M_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
        elif model_to_load == 'K_UNet': 
            model = K_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features) 
        elif model_to_load == 'UNet_2Plus': 
            from UNet_2Plus import UNet_2Plus 
            model = UNet_2Plus(in_channels=in_channels, n_classes=out_channels)
        elif model_to_load == 'UNet_3Plus': 
            from UNet_3Plus import UNet_3Plus
            model = UNet_3Plus(in_channels=in_channels, n_classes=out_channels)
        elif model_to_load == 'MultiResUNet':
            from MultiResUNet import MultiResUNet
            model = MultiResUNet(in_features=32, alpha=1.67, in_channels=in_channels, out_channels=out_channels)

        elif model_to_load == 'CSNet':
            from csnet import CSNet
            model = CSNet(classes=out_channels , channels=in_channels)


        elif model_to_load == 'Self_UNet':
            from SelfONN_Unet import Self_UNet
            model = Self_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SelfUNet_compact':
            from SelfONN_Unet import SelfUNet_compact
            model = SelfUNet_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SelfUNet_super_compact':
            from SelfONN_Unet import SelfUNet_super_compact
            model = SelfUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SuperUNet_super_compact':
            from SelfONN_Unet import SuperUNet_super_compact
            model = SuperUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Super_FPN':
            from SelfONN_Unet import Super_FPN
            model = Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=3, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Attention_Super_FPN':
            from SelfONN_Unet import Attention_Super_FPN
            model = Attention_Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Densenet121 SuperONN_Unet':
            model = SelfONNUnet('densenet121', decoder_attention_type = decoder_attention, in_channels=3, classes=1, q_order = q_order, max_shift = max_shift, activation='logsoftmax')
        elif model_to_load == 'SelfONN_Unet_new':
            model = SelfONNUnet_new('densenet121', decoder_attention_type = decoder_attention, in_channels=3, classes=1, q_order = 3, max_shift = 0, activation=activation)
        elif model_to_load == 'SA_UNet':
            model = SA_UNet(in_channels=in_channels, num_classes=out_channels)
        elif model_to_load == 'ACC_UNet':
            model = ACC_UNet(n_channels=in_channels, n_classes=out_channels, n_filts=U_init_features)
        elif model_to_load == 'CSC_UNet':
            model = CSC_UNet(in_channel=in_channels, out_channel=out_channels, unfolding=unfolding)
        elif model_to_load == 'CSC_UNet_2Plus':
            model = CSCUNetPP(input_channels=in_channels, num_classes=out_channels, unfolding=unfolding)
        elif model_to_load == 'CSC_UNet_3Plus':
            model = CSCUNet3PlusDeepSup(in_channels=in_channels, n_classes=out_channels, unfolding=unfolding)
        elif model_to_load == 'CSCA_UNet':
            model = CSCAUNet(in_channels=in_channels, n_class=out_channels, decay=unfolding)
        elif model_to_load == 'densenet121_CSCA_UNet':
            model = CSCAUNet_DenseNet(in_channels=in_channels, n_class=out_channels, decay=unfolding)
        elif model_to_load == 'densenet121_SelfONN_CSCA_UNet':
            model = SelfONN_CSCAUNet_DenseNet(in_channels=in_channels, n_class=out_channels, decay=unfolding, q=q_order)
        elif model_to_load == 'densenet121_FullSelfONN_CSCA_UNet':
            model = FullSelfONN_CSCAUNet_DenseNet(in_channels=in_channels, n_class=out_channels, decay=unfolding, q=q_order)



    # reset_fn = reset_function_generic 
    # model.apply(reset_fn) 

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)
    return model 