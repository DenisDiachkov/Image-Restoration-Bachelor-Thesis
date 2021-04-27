import os
import albumentations as A
from dataset import ImageDataset
import numpy as np
from PIL import Image
from pqdm.processes import pqdm

PATH_TO_IMAGES = os.path.realpath("dataset/GOPRO_Large_all/")
OUT_PATH_TO_CORRUPTED_IMAGES = os.path.realpath("dataset/GoPro/corrupted")
OUT_PATH_TO_GROUND_TRUTH = os.path.realpath("dataset/GoPro/gt")

BLUR_RATE = 5
NOISE_VARIANCE = (100, 200)
DOWNSCALE_RATE = 4


def process_video(params):
    PATH_TO_IMAGES, OUT_PATH_TO_CORRUPTED_IMAGES, \
    OUT_PATH_TO_GROUND_TRUTH, BLUR_RATE, NOISE_VARIANCE, \
    DOWNSCALE_RATE, video_id, folder, transforms = params
    dataset = ImageDataset(os.path.join(PATH_TO_IMAGES, folder) , "folder")
    images_accumulated = np.zeros((dataset[0].shape[0] // DOWNSCALE_RATE, dataset[0].shape[1] // DOWNSCALE_RATE, 3))
    image_gt = None
    for image_id, image in enumerate(dataset):
        if (image_id % BLUR_RATE == BLUR_RATE // 2):
            image_gt = image[...]
        image = transforms(image=image)["image"]
        images_accumulated += image
        if(image_id % BLUR_RATE == BLUR_RATE - 1):
            images_accumulated /= BLUR_RATE
            images_accumulated = np.clip(images_accumulated, 0.0, 255.0)
            corrupt_path = os.path.join(OUT_PATH_TO_CORRUPTED_IMAGES, f"{video_id:03}", f"{image_id // BLUR_RATE:03}") + '.png'
            gt_path = os.path.join(OUT_PATH_TO_GROUND_TRUTH, f"{video_id:03}", f"{image_id // BLUR_RATE:03}") + '.png'
            if not os.path.exists(os.path.split(corrupt_path)[0]):
                os.makedirs(os.path.split(corrupt_path)[0])
            if not os.path.exists(os.path.split(gt_path)[0]):
                os.makedirs(os.path.split(gt_path)[0])
            Image.fromarray(images_accumulated.astype(np.uint8)).save(corrupt_path)
            Image.fromarray(image_gt).save(gt_path)


if __name__ == '__main__':

    transforms = A.Compose([
        A.Resize(720 // DOWNSCALE_RATE, 1280 // DOWNSCALE_RATE, always_apply=True),
        A.GaussNoise(NOISE_VARIANCE, always_apply=True)
    ])
    args = []
    for video_id, folder in enumerate(sorted(os.listdir(PATH_TO_IMAGES))):
        args.append([
            PATH_TO_IMAGES,
            OUT_PATH_TO_CORRUPTED_IMAGES,
            OUT_PATH_TO_GROUND_TRUTH,
            BLUR_RATE,
            NOISE_VARIANCE,
            DOWNSCALE_RATE,
            video_id, folder, transforms
        ])
    pqdm(args, process_video, n_jobs=8)