import csv


def load_dataset_vin(path):
    data = []
    with open(path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        # ['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max']
        reader.fieldnames = ['Path', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max']
        for row in reader:
            if row['Path'] == "image_id":
                continue
            row["Path"] = row["Path"] + ".png"
            data.append(row)
    return data
