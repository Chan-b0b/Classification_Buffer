from PIL import Image
from datetime import datetime
import pytz
import os
import torchvision.transforms as transforms
import torch

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    
def get_result_folder(result_folder):
    process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
    result_folder = os.path.join(result_folder, process_start_time.strftime("%Y%m%d_%H%M%S"))
    return result_folder

def augment_image(input, basic = True):
    if basic :
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    
    else:
        transform = transforms.Compose([
        transforms.Resize((272,272)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
        
    return transform(input)

def save_checkpoints(file_path, model, epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (datetime.now(), file_path))
    file_path = file_path + f'/model.pt'
    checkpoint = {
        'encoder_state_dict': model.state_dict(),
        'epoch_idx': epoch,
    }
    torch.save(checkpoint, file_path)