import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--zip_file_path', type=str, help="Path of zip file", default = '/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/medsam.zip')
parser.add_argument('--output_dir', type=str,  help="Output directory", default = '/home/user01/aiotlab/dung_paper/groundingLMM/LISAMed/')
args = parser.parse_args()

# zip_file_path = 'speech_modules/data/original_data/SLU/train_data.zip'
with zipfile.ZipFile(args.zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(args.output_dir)