# CNN configuration file 

##### DO NOT EDIT THESE LINES #####
config = {} 
###################################


#### START EDITING FROM HERE ######
config['parentdir'] = 'C:/Users/AnasP/OneDrive/Documents/Pytorch/LatestCodes/CXR_segmentation/'# main directory
config['classes'] = ['non COVID', 'COVID-19']  
config['batch_size']  = 4                                                    # batch size, Change to fit hardware
config['in_channels'] = 1                                                    # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
config['out_channels'] = 2                                                   # Binary output is added so far: use '2' for NLLLoss, or '1' for 'BCELoss' or 'DiceLoss' 
config['input_mean'] = [0.4435]                                   # Dataset mean, provide 3 numbers for RGB images or 1 number for gray scale images
config['input_std'] = [0.2795]                                    # Dataset std, provide 3 numbers for RGB images or 1 number for gray scale images
config['lossType'] = 'NLLLoss'                                               # loss function: 'BCELoss' or 'DiceLoss' or 'NLLLoss'
config['num_folds']  = 1                                                     # number of cross validation folds
config['Resize_h'] = 256                                                     # network input size
config['Resize_w'] = config['Resize_h']         
# config['load_model'] = config['parentdir'] + 'load_model/K_UNet_6712_fold_1.pt'     # specify full path of pretrained model pt file 
config['load_model'] = False                                                 # or set to False to load trained model by train code 
config['model_name'] = 'Resnet50_UnetPlusPlus_infection_seg'                                         # name of trained model .pt file, same name used in train code
config['new_name'] = 'Resnet50_UnetPlusPlus_infection_seg'                                 # specify a new folder name to save test results, 
#                                                                            # else set to False to overwrite test results genertaed by train code
##################  

##################
config['Results_path'] = config['parentdir'] + 'Results Infection Seg'                     # main results file 
if config['new_name']:
    config['save_path'] = config['Results_path'] +'/'+ config['new_name']    # new save path  
else: 
    config['save_path'] = config['Results_path'] +'/'+ config['model_name']  # same save path used for training 
##################





