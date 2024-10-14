import json

# Đường dẫn tới tệp văn bản
file_id = 'p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4'
file_id = 'p10/p10000980/s54935705/6ad819bb-bae74eb9-7b663e90-b8deabd7-57f8054a'
file_id = 'p10/p10000980/s58206436/54affd39-8bf24209-232bac8a-df6c277a-398ee8a5'
file_id = 'p10/p10001401/s51065211/c74ce171-c7c53262-a7d57fa1-ee9a9bea-b5f75cb8'

file_path = f'/home/user01/aiotlab/dung_paper/groundingLMM/dataset/mimic_processed/MIMIC_MedGLaMM_caption/{file_id}.txt'
# Đọc toàn bộ nội dung của tệp
with open(file_path, 'r') as file:
    file_content = file.read()
# Chuyển đổi từ chuỗi JSON thành từ điển Python
data_dict = json.loads(file_content)
# print(data_dict['impression'])

bounding_boxes = []
for impression in data_dict['impression']:
    # print(bounding_boxes, impression)
    disease_name, box = impression['disease']['name'], impression['disease']['box']
    bounding_boxes.append({'label':disease_name, 'bbox': box})
    # print(disease_name)
    # print(disease_name)
    # anatomies = []
    # seg_masks = []
    # height = 1500
    # width  =2250
    # for anatomy in impression['anatomies']:
    #     anatomies.append(anatomy['name_anatomy'])
    #     seg_masks.append(anatomy['anatomy_mask']['segmentation']['counts'])
    # print(anatomies)

import cv2
import matplotlib.pyplot as plt

# Image path and example bounding boxes for diseases
image_path = f'/home/user01/aiotlab/dung_paper/groundingLMM/mimic-cxr-jpg/2.1.0/files/{file_id}.jpg'

image = cv2.imread(image_path)
print(image.shape)
# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)

color = (0, 255, 0)
thickness = 2
font_scale = 4 # Make text a bit smaller
font_thickness = 4  # Thicker text

# Store positions of labels to avoid overlap
used_label_positions = []
# Draw bounding boxes and annotate them with disease labels
for box in bounding_boxes:
    x_min, y_min, x_max, y_max = box["bbox"]
    
    # Draw the bounding box
    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)
    
    class_name = box['label']
    # Get the size of the label text
    label_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Set initial label position
    label_x = x_min
    label_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
    
    # Check if this position overlaps with any previous labels
    while any([abs(label_y - pos[1]) < label_size[1] for pos in used_label_positions]):
        label_y -= label_size[1] + 10  # Move label up to avoid overlap

    # Store this label position
    used_label_positions.append((label_x, label_y))

    # Put the class name text with a red color
    cv2.putText(image_rgb, class_name, (label_x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

plt.axis('off')  # Hide axes
plt.show()
file_id = file_id.replace('/','_')
output_image_path = f'{file_id}.png'
cv2.imwrite(output_image_path, image_rgb)
print(f"Image saved to {output_image_path}")