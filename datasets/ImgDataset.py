# -*- encoding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import tifffile as tiff
from osgeo import gdal, osr

class ImageDataset(Dataset):


    def __init__(self, im_data, tile_size, overlap, transform,
                 channel_first=False,in_path = None,output_folder= None,
                 filename_prefix= None, level1_window=None
                 ):

        if channel_first:
            im_data = np.transpose(im_data, (1, 2, 0))
        self.image = im_data
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tile_coords = self._get_tile_coordinates(im_data)
        self.in_path = in_path

        self.output_folder = output_folder
        self.filename_prefix = filename_prefix
        self.level1_window = level1_window

    def _get_tile_coordinates(self, image):

        stride = self.tile_size - self.overlap
        img_h, img_w, img_c = image.shape
        n_h = int(np.ceil((img_h - self.tile_size) / stride)) + 1
        n_w = int(np.ceil((img_w - self.tile_size) / stride)) + 1
        windows = []
        for i in range(n_h):
            dh = min(i * stride, img_h - self.tile_size)
            for j in range(n_w):
                dw = min(j * stride, img_w - self.tile_size)
                if np.sum(image[dh:dh + self.tile_size, dw:dw + self.tile_size, :]) == 0:
                    continue
                windows.append([dh, dh + self.tile_size, dw, dw + self.tile_size])
        return windows

    def __len__(self):

        return len(self.tile_coords)

    def __getitem__(self, i):

        window = self.tile_coords[i]

        y1, y2, x1, x2 = window
        tile = self.image[y1:y2, x1:x2]
        # print("tile",tile.shape)



        processor = RSImageProcessor(self.in_path,
                                     level1_window=self.level1_window  )
        tif_path = processor.crop_and_save_tile(
            window=window,
            level1_window=self.level1_window,
            output_folder=self.output_folder,
            filename_prefix=self.filename_prefix)


        transformed = self.transform(image=tile)

        tile = transformed['image']

        return {
            'image': tile,
            'window': np.array(window),
            'file_path': tif_path,
        }


class RSImageProcessor:
    def __init__(self, image_path, level1_window=None):

        self.dataset = gdal.Open(image_path)
        if self.dataset is None:
            raise ValueError(f"无法打开影像文件: {image_path}")


        self.image = self.dataset.ReadAsArray()  # (C, H, W)
        self.image = np.transpose(self.image, (1, 2, 0))  # 转换为 (H, W, C)


        self.geo_transform = self.dataset.GetGeoTransform()
        self.projection = self.dataset.GetProjection()
        self.level1_window = level1_window


        if self.image.shape[2] != 15:
            raise ValueError("输入影像波段数不是15个")

    def crop_and_save_tile(self, window, level1_window, output_folder, filename_prefix):

        y1_l2, y2_l2, x1_l2, x2_l2 = window


        y1_l1, y2_l1, x1_l1, x2_l1 = level1_window


        y1_abs = y1_l1 + y1_l2
        y2_abs = y1_l1 + y2_l2
        x1_abs = x1_l1 + x1_l2
        x2_abs = x1_l1 + x2_l2


        if (y1_abs < 0 or y2_abs > self.image.shape[0] or
                x1_abs < 0 or x2_abs > self.image.shape[1]):
            print(f"警告：绝对坐标超出范围: ({y1_abs},{y2_abs},{x1_abs},{x2_abs})")
            print(f"  第一层窗口: {level1_window}")
            print(f"  第二层窗口: {window}")

        abs_window = (y1_abs, y2_abs, x1_abs, x2_abs)


        tile = self.image[y1_l2:y2_l2, x1_l2:x2_l2, :]


        os.makedirs(output_folder, exist_ok=True)


        multiband_path = self._save_multiband_tif(
            tile=tile,
            output_folder=output_folder,
            filename_prefix=filename_prefix,
            geo_transform=self._get_adjusted_geo_transform(abs_window),
            projection=self.projection,
            window = abs_window
        )


        return multiband_path

    def _get_adjusted_geo_transform(self, window):

        y1, _, x1, _ = window
        original_gt = self.geo_transform


        new_upper_left_x = original_gt[0] + x1 * original_gt[1] + y1 * original_gt[2]
        new_upper_left_y = original_gt[3] + x1 * original_gt[4] + y1 * original_gt[5]


        new_gt = (
            new_upper_left_x, original_gt[1], original_gt[2],
            new_upper_left_y, original_gt[4], original_gt[5]
        )

        return new_gt

    def _save_multiband_tif(self, tile, output_folder, filename_prefix,
                            geo_transform, projection, window):

        y1, y2, x1, x2 = window
        fname = f"{filename_prefix}_{y1}_{x1}.tif"

        multiband_path = os.path.join(output_folder, fname)


        height, width, channels = tile.shape


        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            multiband_path,
            width,
            height,
            channels,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_NEEDED']
        )


        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)


        for band_idx in range(channels):
            dataset.GetRasterBand(band_idx + 1).WriteArray(tile[:, :, band_idx])


        dataset.FlushCache()
        dataset = None

        return multiband_path


class Sentinel2Dataset(Dataset):


    def __init__(self, im_data, tile_size, overlap, transform, channel_first=False):

        if channel_first:
            im_data = np.transpose(im_data, (1, 2, 0))
        self.image = im_data
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tile_coords = self._get_tile_coordinates(im_data)

    def _get_tile_coordinates(self, image):

        stride = self.tile_size - self.overlap
        img_h, img_w, img_c = image.shape
        n_h = int(np.ceil((img_h - self.tile_size) / stride)) + 1
        n_w = int(np.ceil((img_w - self.tile_size) / stride)) + 1
        windows = []
        for i in range(n_h):
            dh = min(i * stride, img_h - self.tile_size)
            for j in range(n_w):
                dw = min(j * stride, img_w - self.tile_size)
                if np.sum(image[dh:dh + self.tile_size, dw:dw + self.tile_size, :]) == 0:
                    continue
                windows.append([dh, dh + self.tile_size, dw, dw + self.tile_size])
        return windows

    def __len__(self):

        return len(self.tile_coords)

    def __getitem__(self, i):

        window = self.tile_coords[i]
        y1, y2, x1, x2 = window
        im_data = self.image[y1:y2, x1:x2]
        im_data = self.transform(im_data)
        return {
            'image': torch.from_numpy(im_data).to(torch.float32),
            'window': np.array(window)
        }

        