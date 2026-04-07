# @Version : 1.0
# @Author  : 张公子
# @File    : 2_DaTa_Enhance_TY.py
# @Time    : 2025/6/20 15:14
import os
import glob
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show
import numpy as np

def get_image_paths(folder):

    return glob.glob(os.path.join(folder, '*.tif'))

def create_read_img(filename):

    with rasterio.open(filename) as src:

        image = src.read()
        print("Label shape:", image.shape)

        profile = src.profile


        output_dir = r'/root/autodl-tmp/Project/rs-segment-pytorch-main/GuoChengWenJian/SY_CaiJian/ZengQiang_shuchu'


        base_name = os.path.splitext(os.path.basename(filename))[0]


        out_h = np.flip(image, axis=2)
        save_transformed_image(out_h, profile, os.path.join(output_dir, f"{base_name[0]}h_DEM.tif"))


        out_w = np.flip(image, axis=1)
        save_transformed_image(out_w, profile, os.path.join(output_dir, f"{base_name[0]}w_DEM.tif"))


        # out_90 = np.rot90(image, k=1, axes=(1, 2))
        # save_transformed_image(out_90, profile, filename[:-4] + '_90.tif')
        #

        # out_180 = np.rot90(image, k=2, axes=(1, 2))
        # save_transformed_image(out_180, profile, filename[:-4] + '_180.tif')
        #

        # out_270 = np.rot90(image, k=3, axes=(1, 2))
        # save_transformed_image(out_270, profile, filename[:-4] + '_270.tif')
        #

        # image_brightened = image * 1.5
        # save_transformed_image(image_brightened, profile, filename[:-4] + '_brighter.tif')
        #

        # image_colored = image * 1.5
        # save_transformed_image(image_colored, profile, filename[:-4] + '_color.tif')
        #

        # image_contrasted = image * 1.5
        # save_transformed_image(image_contrasted, profile, filename[:-4] + '_contrast.tif')
        #

        # image_sharped = image * 3.0
        # save_transformed_image(image_sharped, profile, filename[:-4] + '_sharp.tif')

def save_transformed_image(image, profile, output_filename):

    profile.update(dtype=image.dtype)

    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(image)

if __name__ == '__main__':

    img_path = r'/root/autodl-tmp/Project/rs-segment-pytorch-main/GuoChengWenJian/SY_CaiJian/label/HuiZong'

    imgs = get_image_paths(img_path)

    for i in imgs:

        create_read_img(i)