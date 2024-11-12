import os
from glob import glob
from tqdm import tqdm
import json

import numpy as np
import cv2
from PIL import Image
import imagehash

from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor


def load_original_data():
    path = "/mnt/12T/01_hieu/VLM/data/xray14_resized_224x224/cxr"
    image_paths = glob(os.path.join(path, "images_*/images", "*"))
    images = {}
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('L')
        org_image_hash = str(imagehash.phash(image))
        images[org_image_hash] = {
            "image_path": image_path,
            "image": image
        }
    return images


def compute_hashes(start_index, stop_index, train_dataset):
    local_hashes = {}
    for idx in tqdm(range(start_index, stop_index)):
        image = train_dataset[idx]
        image_hash = str(imagehash.phash(image))
        local_hashes[idx] = {
            "image": image,
            "image_hash": image_hash
        }
    return local_hashes


def get_chestx_hash(train_dataset):
    train_dataset = train_dataset["image"]
    hash_chestx = {}
    num_threads = 20  # Number of threads
    total_images = len(train_dataset)
    batch_size = total_images // num_threads  # Divide dataset size by number of threads

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        # Submit tasks to the thread pool
        for i in range(num_threads):
            start_index = i * batch_size
            # Ensure the last thread processes all remaining images
            stop_index = total_images if i == num_threads - 1 else (i + 1) * batch_size
            futures.append(executor.submit(compute_hashes, start_index, stop_index, train_dataset))

        # Collect results as threads complete
        for future in tqdm(futures, total=len(futures)):
            result = future.result()
            hash_chestx.update(result)

    # Now hash_chestx contains all the image hashes
    return hash_chestx


def main():
    dirpath_dest = "tmp"
    os.makedirs(dirpath_dest, exist_ok=True)
    train_dataset = load_dataset("chestx", split="train")
    test_dataset = load_dataset("chestx", split="test")
    org_images = load_original_data()

    # hash_chestx = {}
    # for image_index, batch in tqdm(enumerate(train_dataset), total=len(train_dataset)):
    #     image = batch["image"]
    #     image_hash_image = str(imagehash.phash(image))
    #     hash_chestx[image_index] = image_hash_image

    hash_chestx = get_chestx_hash(test_dataset)

    mapping_data = []
    for image_index, data in tqdm(hash_chestx.items()):
        image = data["image"]
        image_hash = data["image_hash"]
        if image_hash in org_images.keys():
            image_np = np.array(image)
            org_image_np = np.array(org_images[image_hash]["image"])
            concat_image = np.concatenate((image_np, org_image_np), axis=1)
            cv2.imwrite(os.path.join(dirpath_dest, os.path.basename(org_images[image_hash]["image_path"]).replace(".", f"_{image_index}.")), concat_image)
            mapping_data.append(
                {
                    image_index: org_images[image_hash]["image_path"]
                }
            )
    mapping_data = {"mapping": mapping_data}
    with open("mapping_chestx_test.json", "w") as f:
        json.dump(mapping_data, f)


if __name__ == "__main__":
    main()
