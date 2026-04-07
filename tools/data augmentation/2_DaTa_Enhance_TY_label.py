# @Version : 1.0
# @Author  : 张公子
# @File    : 2_DaTa_Enhance_TY_label.py
# @Time    : 2025/6/27 21:07
import os
import glob
import rasterio
import numpy as np


def get_image_paths(folder):
    return glob.glob(os.path.join(folder, '*.tif'))


def create_read_img(filename):
    with rasterio.open(filename) as src:
        image = src.read()
        profile = src.profile.copy()
        print("Label shape:", image.shape)

    output_dir = r'/root/autodl-tmp/Project/rs-segment-pytorch-main/GuoChengWenJian/SY_CaiJian/ZengQiang_shuchu'
    base_name = os.path.splitext(os.path.basename(filename))[0]

    out_h = np.flip(image, axis=2)
    save_transformed_image(out_h, profile, os.path.join(output_dir, f"{base_name[0]}h_DEM.tif"))

    out_w = np.flip(image, axis=1)
    save_transformed_image(out_w, profile, os.path.join(output_dir, f"{base_name[0]}w_DEM.tif"))


def save_transformed_image(image, profile, output_filename):
    profile.update(dtype=image.dtype)
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(image)


if __name__ == '__main__':
    img_path = r'/root/autodl-tmp/Project/rs-segment-pytorch-main/GuoChengWenJian/SY_CaiJian/label/HuiZong'
    imgs = get_image_paths(img_path)
    for i in imgs:
        create_read_img(i)

