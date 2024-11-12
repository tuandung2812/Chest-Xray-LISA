import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


def main():
    filepath_test = "/mnt/12T/02_duong/data-center/Data-Large-MultiModal-Models/MIMIC_CXR/MIMIC_MedGLaMM_caption_v3/p10/p10000935/s51178377/9b314ad7-fbcb0422-6db62dfc-732858d0-a5527d8b.json"
    filepath_image = "/mnt/12T/01_hieu/VLM/data/2019.MIMIC-CXR-JPG/2.0.0/files/p10/p10000935/s51178377/9b314ad7-fbcb0422-6db62dfc-732858d0-a5527d8b.jpg"
    with open(filepath_test) as file:
        data_impression = json.load(file)
    img = mpimg.imread(filepath_image)
    for impression in data_impression["impression"]:
        polygons  = impression["anatomy"]["anatomy_polygons"]
        disease_name = impression["disease"]["disease_name"]
        anatomy_name = impression["anatomy"]["anatomy_name"]
        if polygons is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.set_title("Anatomy Visualization")
            
            points = list(zip(polygons[0][::2], polygons[0][1::2]))
            
            # Create and add the polygon to the plot
            poly = patches.Polygon(points, closed=True, fill=True, color='blue', alpha=0.5)
            ax.add_patch(poly)
            
            ax.text(points[0][0], points[0][1], f"{disease_name} {anatomy_name}", color='red', fontsize=16, ha='center')

            plt.axis('off')
            plt.savefig(f"anatomy_visualization_{disease_name}_{anatomy_name}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
