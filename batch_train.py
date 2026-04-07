import os

# Path to the config file and training script
config_file = 'config.py'
training_script = 'TrainUnet.py'

# Function to update the config file
def update_config_file(model_name, decoder_name):
    with open(config_file, 'r') as file:
        config_lines = file.readlines()

    # Update model_to_load and model_name in config.py
    new_model_to_load = f"{model_name}*{decoder_name}"
    new_model_name = f"THA_Flip_{model_name}_{decoder_name}"

    # Iterate through config file and replace required fields
    with open(config_file, 'w') as file:
        for line in config_lines:
            if line.startswith("config['model_to_load']"):
                line = f"config['model_to_load'] = '{new_model_to_load}'\n"
            elif line.startswith("config['model_name']"):
                line = f"config['model_name'] = '{new_model_name}'\n"
            file.write(line)

    print(f"Updated config.py with model_to_load: {new_model_to_load} and model_name: {new_model_name}")

# Function to run the training
def run_training():
    os.system(f'python {training_script}')
    print(f"Training completed for the current model.")

# List of models to train (encoder, decoder pairs)
model_list = [
    # Model suggestions from the assistant
    ('efficientnet-b4', 'DeepLabV3Plus'),    # Suggested 1: DeepLabV3+ with EfficientNet-B4
    ('timm-resnest50d', 'FPN'),                   # Suggested 2: FPN with ResNeSt
    ('se_resnet50', 'UnetPlusPlus'),          # Suggested 3: UNet++ with SE-Net
    ('densenet121', 'PSPNet'),               # Suggested 4: PSPNet with DenseNet
    ('timm-res2net50_26w_4s', 'MAnet'),           # Suggested 5: MAnet with Res2Net
    ('resnet50', 'Unet'),
    ('tu-tf_efficientnet_b6', 'Unet'),
    # ('tu-tf_efficientnetv2_l', 'MAnet'),
    # ('tu-tf_efficientnetv2_m', 'MAnet'),
    #('tu-tf_efficientnet_b6', 'SelfONN_MAnet'),
    # ('timm-resnest200e', 'Unet'),
    # ('tu-maxvit_base_tf_224', 'MAnet'),
    # ('tu-maxvit_large_tf_384', 'MAnet'),

    
    # Add any other models you want to train below
    # Example:
    # ('resnet50', 'Unet'),
    # ('efficientnet-b3', 'FPN'),
]

# Loop through the list of models and run training for each
for model_name, decoder_name in model_list:
    update_config_file(model_name, decoder_name)
    run_training()
