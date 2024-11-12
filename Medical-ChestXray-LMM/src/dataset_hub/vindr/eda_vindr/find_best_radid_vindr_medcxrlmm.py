import sys
sys.path.append(".")

import pandas as pd


from dataset_hub.vindr.config_vindr.config_vindr_medcxrlmm import *


def find_based_counts():
    df = pd.read_csv(FILEPATH_VINDR_MEDCXRLMM_CAPTION_MEDATADATA)
    rad_ids = df["rad_id"].unique()
    disease_names = df["disease_name"].unique()

    data_records = []
    for rad_id in rad_ids:
        data_radid = df[df["rad_id"] == rad_id]
        for disease_name in disease_names:
            data_disease = data_radid[data_radid["disease_name"] == disease_name]
            data_disease = data_disease["filepath_image"].unique()
            record = {
                "rad_id": rad_id,
                "disease_name": disease_name,
                "count": len(data_disease)
            }
            data_records.append(record)
    df = pd.DataFrame(data_records)


    import matplotlib.pyplot as plt
    import seaborn as sns

    for disease in disease_names:
        disease_data = df[df['disease_name'] == disease]
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x='rad_id', y='count', data=disease_data, palette='viridis')
        plt.title(f'Count of {disease} by Rad ID')
        plt.xlabel('Rad ID')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if type(disease) == float:
            disease = 'No Finding'
        plt.savefig(f"{disease.replace('/', '_').replace(' ', '_')}.png")
        plt.close() 

    # Get unique rad_ids and diseases
    rad_ids = df['rad_id'].unique()
    diseases = df['disease_name'].unique()

    # Create a grid of subplots
    num_rad_ids = len(rad_ids)
    num_diseases = len(diseases)
    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_rad_ids, num_diseases, figsize=(30, 3 * num_rad_ids), sharey=True)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Iterate over each rad_id and disease to create subplots
    for i, rad_id in enumerate(rad_ids):
        for j, disease in enumerate(diseases):
            # Filter data for the current rad_id and disease
            disease_data = df[(df['rad_id'] == rad_id) & (df['disease_name'] == disease)]
            
            if not disease_data.empty:
                axes[i, j].bar(disease_data['disease_name'], disease_data['count'], color='blue')
                axes[i, j].set_title(f'{rad_id}: {disease}')
                axes[i, j].set_xlabel('Disease')
                axes[i, j].set_ylabel('Count')
            else:
                axes[i, j].bar([disease], [0], color='lightgrey')
                axes[i, j].set_title(f'{rad_id}: {disease} (No data)')
                axes[i, j].set_ylabel('Count')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save the figure directly to the current directory
    fig.savefig("All_Rad_IDs_Diseases.png")


if __name__ == '__main__':
    find_based_counts()
