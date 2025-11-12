from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

# Package for dataloader
from Utility.DataLoader.data_loader import RescaleT
from Utility.DataLoader.data_loader import ToTensorLab
from Utility.DataLoader.data_loader import SalObjDataset

from Utility.Model.u2net import U2NET # full size version 173.6 MB
from Utility.Model.u2net import U2NETP # small version u2net 4.7 MB

# Library for system and files manipulation
import sys
import os

# Define new project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def normPRED(d):
    """
    Normalizes a PyTorch tensor to the range [0, 1] using min-max scaling.

    Args:
        d (torch.Tensor): The input tensor to be normalized.

    Returns:
        torch.Tensor: The normalized tensor with values scaled between 0 and 1.
    """
    # Find the maximum value in the tensor
    ma = torch.max(d)
    # Find the minimum value in the tensor
    mi = torch.min(d)

    # Apply the min-max normalization formula: (value - min) / (max - min)
    dn = (d - mi) / (ma - mi)

    return dn

def save_output(image_name, pred, d_dir):
    """
    Saves a model's prediction tensor as a PNG image.

    The prediction is first converted to a PIL Image, resized to match the
    dimensions of the original input image, and then saved to the specified
    directory with a .png extension.

    Args:
        image_name (str): The file path to the *original* input image. This is
                          used to get the original dimensions and the output filename.
        pred (torch.Tensor): The model's prediction tensor. Expected to be
                             a 2D or 3D tensor (e.g., [1, H, W] or [H, W])
                             with values that can be normalized (ideally [0, 1]).
        d_dir (str): The directory path where the output PNG image will be saved.
    """
    
    # Squeeze the tensor to remove batch or channel dimensions (e.g., [1, H, W] -> [H, W])
    predict = pred.squeeze()
    
    # Move the tensor to the CPU, detach it from the computation graph, and convert to NumPy
    predict_np = predict.cpu().data.numpy()

    # Scale the prediction from [0, 1] to [0, 255] and convert to a PIL Image
    # .convert('RGB') ensures the saved image is 3-channel, even if the mask is 1-channel
    im = Image.fromarray(predict_np * 255).convert('RGB')

    # Get the filename (e.g., "my_image.jpg") from the full path
    img_name = image_name.split(os.sep)[-1]
    
    # Read the original image to get its shape for resizing
    # Note: scikit-image io.imread loads as (height, width, channels)
    image = io.imread(image_name)

    # Resize the prediction mask to match the original image's width and height
    # PIL's resize expects (width, height)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    # --- Reconstruct the filename without the original extension ---
    # This block handles filenames that might contain dots (e.g., "img.v2.jpg")
    # A simpler way for most cases would be: imidx = os.path.splitext(img_name)[0]
    
    aaa = img_name.split(".")  # e.g., "img.v2.jpg" -> ["img", "v2", "jpg"]
    bbb = aaa[0:-1]           # e.g., ["img", "v2"]
    imidx = bbb[0]            # e.g., "img"
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]  # e.g., "img" + "." + "v2" -> "img.v2"
    # --- End of filename reconstruction ---

    # Save the resized PIL image as a PNG file in the destination directory
    imo.save(d_dir + imidx + '.png')

def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    input_dir ="D:/Pill_Identification/Data/PILL_JPG_2025"
    prediction_dir="D:/Pill_Identification/Data/BackgroundRemoveData/"
    model_dir = "D:/Pill_Identification/background_removal_DL/saved_models/u2net/u2net.pth"

    img_name_list = []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img_path = os.path.join(root, filename)
                img_name_list.append(img_path)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()