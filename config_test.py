# CNN test configuration file 

##### DO NOT EDIT THESE LINES #####
config = {}
###################################

#### START EDITING FROM HERE ######
config['parentdir'] = '/content/'                                                        # main directory
config['ONN'] = True                                                                     # set to 'True' if you are using ONN
config['batch_size'] = 8                                                                 # batch size, Change to fit hardware
config['in_channels'] = 3                                                                # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
config['out_channels'] = 1                                                               # '1' for binary class losses: 'BCELoss' or 'DiceLoss', 'number of classes' for multi-class losses
config['input_mean'] = [0.0, 0.4878, 0.5057]                                             # provide 3 numbers for RGB images or 1 number for gray scale images in list format 
config['input_std'] = [0.0, 0.1971, 0.2013]                                              # provide 3 numbers for RGB images or 1 number for gray scale images      
config['num_folds'] = 5                                                                  # number of folds
config['class_weights'] = None                                                           # class weights for multi class masks, default: none
config['lossType'] = 'DiceLoss'    
config['Resize_h'] = 256                                                                 # network input size
config['Resize_w'] = config['Resize_h']         
# config['load_model'] = '/content/drive/Shareddrives/Rusab/Coronary_Angiogram/Results/MixTrain_densenet121*SelfONN_ResUnet/MixTrain_densenet121*SelfONN_ResUnet_fold_3.pt'        # specify full path of pretrained model pt file 
config['load_model'] = False                                                             # specify path of pretrained model wieghts or set to False to train from scratch  
config['Test_Mask'] = True                                                               # set to true if you have the test masks, to compute Evaluation metricies over test set
config['model_to_load'] = 'densenet201*UnetPlusPlus'                                     # check 'available_models.txt' 
config['model_name'] = 'FinalOldMask_densenet201*UnetPlusPlus'            # choose a unique name for result folder 
config['decoder_attention'] = 'scse'                                                   #decoder attention type
config['seg_threshold'] = 0.5                                                           # Segmentation Threshold (Default 0.5)
config['Results_path'] = "/content/drive/MyDrive/Colab_stuff/Coronary_Angiogram/"                     # main results file 
config['save_path'] = config['Results_path'] +'/'+ config['model_name']       # save path 
config['generated_masks']  =  '/content/drive/MyDrive/Colab_stuff/Coronary_Angiogram/' + 'Genereated_mask'         # path to save generated_masks for test set 
config['new_name'] = 'FinalOldMask_densenet201*UnetPlusPlus'                                                                 #specify a new folder name to save test results, 
config['fold_to_run'] = [1,5] # define as [] to loop through all folds, or specify start and end folds i.e. [3, 5] or [5, 5]     # else set to False to overwrite test results genertaed by train code
##################  

##################
config['generated_masks']  =  "/content/drive/MyDrive/Coronary_Angiogram/" + 'Genereated_mask'        # path to save generated_masks for test set 
config['Results_path'] = "/content/drive/MyDrive/Coronary_Angiogram/" + "Results"                      # main results file 
if config['new_name']:
    config['save_path'] = config['Results_path'] +'/'+ config['new_name']                                      # new save path  
else: 
    config['save_path'] = config['Results_path'] +'/'+ config['model_name']                                    # same save path used for training 
##################
