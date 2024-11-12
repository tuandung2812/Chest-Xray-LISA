from datasets import load_dataset


train_dataset = load_dataset("chestx", split="train")
test_dataset = load_dataset("chestx", split="test")

print(train_dataset)
for batch in train_dataset:
    image = batch["image"]  # (224, 224)
    pathols = batch["pathols"]  # 14 pathology classification
    structs = batch["structs"]  # (14 * 224) anatomical segmentation
    break