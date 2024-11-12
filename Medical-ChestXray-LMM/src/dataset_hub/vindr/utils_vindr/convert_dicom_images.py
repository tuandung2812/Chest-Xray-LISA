import os
import glob

import cv2
import numpy as np
import pydicom

INPUT_PATH = "/mnt/12T/02_duong/data-center/VinDr/"


def convert_dicom_to_png(type='train', use_8bit=False, div=1):
    files = glob.glob(INPUT_PATH + '{}/*.dicom'.format(type))
    if use_8bit is False:
        if div != 1:
            out_folder = INPUT_PATH + '{}_png_16bit_div_{}/'.format(type, div)
        else:
            out_folder = INPUT_PATH + '{}_png_16bit/'.format(type)
    else:
        if div != 1:
            out_folder = INPUT_PATH + '{}_png_div_{}/'.format(type, div)
        else:
            out_folder = INPUT_PATH + '{}_png/'.format(type)

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for f in files:
        id = os.path.basename(f)[:-6]
        print('Go for: {}'.format(id))
        out_path = out_folder + '{}.png'.format(id)
        if os.path.isfile(out_path):
            continue
        img = read_xray(f, use_8bit=use_8bit, rescale_times=div)
        print(img.shape, img.min(), img.max(), img.dtype)
        cv2.imwrite(out_folder + '{}.png'.format(id), img)


def read_xray(path, voi_lut=True, fix_monochrome=True, use_8bit=True, rescale_times=None):
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float64)
    if rescale_times:
        data = cv2.resize(data, (data.shape[1] // rescale_times, data.shape[0] // rescale_times), interpolation=cv2.INTER_CUBIC)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)

    if use_8bit is True:
        data = (data * 255).astype(np.uint8)
    else:
        data = (data * 65535).astype(np.uint16)

    return data


def main():
    convert_dicom_to_png("train")
    convert_dicom_to_png("test")


if __name__ == "__main__":
    main()
