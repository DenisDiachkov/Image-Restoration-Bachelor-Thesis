import numpy as np
from PIL import Image
import torch
from pytorch_lightning.metrics.functional import psnr, ssim
import torch.nn.functional as nnf
import os


PATH_TO_CORRUPTED_IMAGES = os.path.realpath("dataset/GoPro/corrupted")
PATH_TO_GROUND_TRUTH = os.path.realpath("dataset/GoPro/gt")

N = 0
total_psnr = 0
total_ssim = 0

for video_id, (corrupted_video_name, gt_video_name) in enumerate(zip(
    sorted(os.listdir(PATH_TO_CORRUPTED_IMAGES)), 
    sorted(os.listdir(PATH_TO_GROUND_TRUTH))
)):
    corrupt_path = os.path.join(PATH_TO_CORRUPTED_IMAGES, corrupted_video_name)
    gt_path = os.path.join(PATH_TO_GROUND_TRUTH, gt_video_name)

    for image_id, (corrupt_image_name, gt_image_name) in enumerate(zip(
        sorted(os.listdir(corrupt_path)), 
        sorted(os.listdir(gt_path))
    )):
        corrupted_image = nnf.interpolate(
            torch.from_numpy(np.array(Image.open(os.path.join(corrupt_path, corrupt_image_name)), dtype=np.float32)).cuda().unsqueeze(0).permute(0, 3, 1, 2), 
            size=(720, 1280), mode='bilinear'
        )
        gt_image = torch.from_numpy(np.array(Image.open(os.path.join(gt_path, gt_image_name)), dtype=np.float32)).cuda().permute(2,0,1).unsqueeze(0)

        total_psnr += psnr(corrupted_image, gt_image, data_range=255.0)
        total_ssim += ssim(corrupted_image, gt_image, data_range=255.0)
        N += 1

        print(os.path.join(corrupt_path, corrupt_image_name))

print(total_psnr / N)   # tensor(18.5913, device='cuda:0') 
print(total_ssim / N)   # tensor(0.7343, device='cuda:0')
