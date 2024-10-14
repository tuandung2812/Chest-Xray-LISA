import os
import argparse
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm
def dicom_to_jpg(input_dir, output_dir):
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt qua tất cả các file trong thư mục đầu vào
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(".dicom"):
                dicom_path = os.path.join(root, file)
                try:
                    # Đọc file DICOM
                    dicom_file = pydicom.dcmread(dicom_path)
                    image_array = dicom_file.pixel_array
                    
                    # Chuẩn hóa hình ảnh
                    image_scaled = (np.maximum(image_array, 0) / image_array.max()) * 255.0
                    image_scaled = np.uint8(image_scaled)

                    # Tạo ảnh JPG và lưu vào thư mục đầu ra
                    image = Image.fromarray(image_scaled)
                    output_file_name = os.path.splitext(file)[0] + '.jpg'
                    output_path = os.path.join(output_dir, output_file_name)
                    image.save(output_path)
                    # print(f"Đã chuyển đổi {dicom_path} thành {output_path}")
                except Exception as e:
                    print(f"Lỗi khi xử lý {dicom_path}: {e}")

if __name__ == "__main__":
    # Thiết lập argparse để nhận các tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Chuyển đổi tất cả các file DICOM trong thư mục thành JPG.")
    parser.add_argument('--input_dir', type=str, help="Thư mục chứa các file DICOM đầu vào", default = '/home/user01/aiotlab/dung_paper/groundingLMM/vinbigdata/train/')
    parser.add_argument('--output_dir', type=str, help="Thư mục lưu trữ các file JPG đầu ra", default= '/home/user01/aiotlab/dung_paper/groundingLMM/vinbigdata/jpg/train/')
    
    args = parser.parse_args()
    print(args)
    
    # Gọi hàm để chuyển đổi các file DICOM sang JPG
    dicom_to_jpg(args.input_dir, args.output_dir)