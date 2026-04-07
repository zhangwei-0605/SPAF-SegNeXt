# -*- encoding: utf-8 -*-

import os
import cv2
import random 
import shutil
import numpy as np
from tqdm import tqdm
import os.path as osp
import pandas as pd
from osgeo import gdal
# import palettable
import matplotlib.pyplot as plt
from collections import defaultdict


Class2Id = {
    'background': 0,
    'budiling': 1,
    'road': 2,
    'forest': 3,
    'grass': 4,
    'farmland': 5,
    'water': 6,
    'bareland': 7,
}

Id2Class = {
    0: 'background',
    1: 'budiling',
    2: 'road',
    3: 'forest',
    4: 'glass',
    5: 'farmland',
    6: 'water',
    7: 'bareland',
}

colors = [
    [0, 0, 0],
    [0, 255, 255],
    [0, 128, 0],
    [0, 255, 0],
    [0, 128, 0],
    [0, 0, 255],
    [255, 0, 0],
    [128, 128, 0],
]


def read_tiff(path):
    raster = gdal.Open(path, gdal.GA_ReadOnly)
    im_data = raster.ReadAsArray(0, 0, raster.RasterXSize, raster.RasterYSize)

    if im_data.ndim == 2:
        im_data = im_data[np.newaxis, :, :]

    del raster
    return im_data


def percentage_truncation(im_data, lower_percent=0.001, higher_percent=99.999, per_channel=True):
    if per_channel:
        out = np.zeros_like(im_data, dtype=np.uint8)
        for i in range(im_data.shape[2]):
            a = 0
            b = 255
            c = np.percentile(im_data[:, :, i], lower_percent)
            d = np.percentile(im_data[:, :, i], higher_percent)
            if (d - c) == 0:
                out[:, :, i] = im_data[:, :, i]
            else:
                t = a + (im_data[:, :, i] - c) * (b - a) / (d - c)
                t = np.clip(t, a, b)
                out[:, :, i] = t
    else:
        a = 0
        b = 255
        c = np.percentile(im_data, lower_percent)
        d = np.percentile(im_data, higher_percent)
        out = a + (im_data - c) * (b - a) / (d - c)
        out = np.clip(out, a, b).astype(np.uint8)
    return out

def get_fid(dir, ext):
    files = os.listdir(dir)
    fids = []
    for f in files:
        if f.endswith(ext):
            fids.append(os.path.splitext(f)[0])
    return fids

def unique_folder(data_dir):
    fids = get_fid(data_dir, '.tif')
    fnum = len(fids)
    n_class = len(Class2Id.items())
    counts = np.zeros(n_class)
    for fid in tqdm(fids):
        label = cv2.imread(osp.join(data_dir, fid+'.tif'), cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        counts += _count

    total = np.sum(counts)
    assert total == fnum*256*256

    ratios = counts / total

    print('{:<10} | {:<10}'.format('class', 'ratio(%)'))
    for i in range(n_class):
        print('{:<10} | {:<10f}'.format(Id2Class[i], ratios[i]*100))

def compute_mean_std(data_dirs):
    paths = []
    for _dir in data_dirs:
        fids = get_fid(_dir, '.tif')
        paths.extend([osp.join(_dir, fid+'.tif') for fid in fids])
    fnum = len(paths)
    print("samples:", len(paths))

    images = []
    for p in tqdm(paths):
        img = cv2.imread(p, cv2.IMREAD_LOAD_GDAL)
        img = img[:, :, :, np.newaxis]
        images.append(img)
    images = np.concatenate(images, axis=3).astype(np.float32) / 255.

    means, stdevs = [], []
    for i in tqdm(range(3)):
        pixels = images[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print('mean:', means)
    print('std:', stdevs)



def calculate_mean_std_hyperspectral(data_dir):
    from osgeo import gdal
    channels = 11

    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    print("images:", len(filepaths))

    pixels = 0
    sumup = np.zeros(channels, np.float64)
    for i, path in enumerate(tqdm(filepaths)):
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        img_w = raster.RasterXSize
        img_h = raster.RasterYSize
        im_bands = raster.RasterCount
        assert im_bands == channels
        im_data = raster.ReadAsArray(0, 0, img_w, img_h)
        im_data = im_data.reshape((im_bands, -1))
        _sum = np.sum(im_data, axis=1)
        sumup += _sum
        pixels += im_data.shape[1]
        del raster
    means = sumup / pixels

    vars = np.zeros(channels, np.float64)
    for i, path in enumerate(tqdm(filepaths)):
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        img_w = raster.RasterXSize
        img_h = raster.RasterYSize
        im_bands = raster.RasterCount
        assert im_bands == channels
        im_data = raster.ReadAsArray(0, 0, img_w, img_h)
        im_data = im_data.reshape((im_bands, -1))
        _var = np.square(im_data - means.reshape((-1, 1)))
        vars += np.sum(_var, axis=1)
        del raster
    stds = np.sqrt(vars / pixels)

    means = list(means.round(3))
    stds = list(stds.round(3))

    print('mean:', means)
    print('std:', stds)



def calculate_mean_std_hyperspectral2(data_dir):
    from osgeo import gdal

    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    print("images:", len(filepaths))

    images = []
    for path in tqdm(filepaths):
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        img_w = raster.RasterXSize
        img_h = raster.RasterYSize
        im_bands = raster.RasterCount
        im_data = raster.ReadAsArray(0, 0, img_w, img_h)
        im_data = im_data.transpose((1, 2, 0)).reshape([-1, im_bands])
        images.append(im_data)
        del raster

    images = np.concatenate(images, axis=0).astype(np.float32)

    means, stdevs = [], []
    for i in tqdm(range(im_bands)):
        pixels = images[:, i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print('mean:', means)
    print('std:', stdevs)


def plot_sample_proportion(dataset_dir, save_dir):
    all_files = [osp.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.tif')]
    total_num = len(all_files)

    data = np.zeros(10)
    for img_path in tqdm(all_files):
        label = cv2.imread(img_path.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)
        vals, counts = np.unique(label, return_counts=True)
        for i, v in enumerate(vals):
            assert v != 0
            data[v-1] += counts[i]

    dpi = 96
    plt.figure(figsize=(1280/dpi, 1280/dpi), dpi=dpi)
    patches, texts, autotexts = plt.pie(x=data, labels=list(Class2Id.keys()), autopct='%.2f%%')
    plt.legend(patches, list(Class2Id.keys()), loc="lower left", bbox_to_anchor=(0, 0, 0.5, 1))
    plt.savefig(osp.join(save_dir, 'sample_proportion.png'))


def sample_proportion(dataset_dirs):

    def softmax(x):
        row_max = x.max()
        row_max = row_max.reshape(-1, 1)
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    def logxy(x, y):
        x = x * np.ones_like(y, dtype=y.dtype)
        return np.log(y) / np.log(x)

    n_class = 10
    label_files = []
    for url in dataset_dirs:
        label_files.extend(sorted([os.path.join(url, f) for f in os.listdir(url) if f.endswith('.png')]))
    total_num = len(label_files)

    class_count = np.zeros(n_class, dtype=np.float64)
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label - 1
        _count = np.bincount(label.flatten(), minlength=n_class)
        class_count += _count

    base_prob = np.sum(class_count) / class_count
    base_prob = np.log2(base_prob)
    base_prob = softmax(base_prob)

    sampling_weights = []
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label - 1
        _count = np.bincount(label.flatten(), minlength=n_class)
        _prob = _count * base_prob
        _prob = np.sum(_prob) / label.size
        sampling_weights.append(_prob)

    sampling_weights = np.array(sampling_weights)
    print(sampling_weights.max())
    print(np.sum(sampling_weights > 0.001))
    print(np.sum(sampling_weights > 0.005))
    print(np.sum(sampling_weights > 0.01))
    print(np.sum(sampling_weights > 0.05))
    print(np.sum(sampling_weights > 0.1))


def check_class(dataset_dir, checkout_dir):
    classlist = list(Class2Id.keys())

    def clustering(all_files, cls_names=[]):
        class_sets = defaultdict(list)
        for img_path in tqdm(all_files):
            label = cv2.imread(img_path.replace('/images/', '/labels/'), cv2.IMREAD_GRAYSCALE)
            vals, counts = np.unique(label, return_counts=True)
            for name in cls_names:
                cls_id = Class2Id[name]
                if cls_id in vals:
                    class_sets[name].append(img_path)
        return class_sets

    _path = osp.join(dataset_dir, 'images')
    all_data = [osp.join(_path, f) for f in os.listdir(_path) if f.endswith('.tif')]
    total_num = len(all_data)
    class_sets = clustering(all_data, classlist)

    print('> check:')
    for cname in classlist:
        check_class_dir = osp.join(checkout_dir, cname)
        os.makedirs(check_class_dir, exist_ok=True)
        for f in tqdm(class_sets[cname]):
            img = cv2.imread(f)
            label = cv2.imread(f.replace('/images/', '/labels/'), cv2.IMREAD_GRAYSCALE)
            cls_id = Class2Id[cname]
            cls_mask = np.where(label == cls_id, cls_id, 0)
            oths_cls_mask = np.where(label == cls_id, 0, label)
            n_class = len(Class2Id.items())
            img_cls_mask = randering_mask(img, cls_mask, n_class, colors, alpha=0.5, beta=0.5)
            img_oths_cls_mask = randering_mask(img, oths_cls_mask, n_class, colors, alpha=0.5, beta=0.5)
            res = np.concatenate([img, img_cls_mask, img_oths_cls_mask], axis=1)
            cv2.imwrite(osp.join(check_class_dir, osp.split(f)[-1]), res)

    print('total samples:', total_num)
    for cname in classlist:
        print('{:<20}'.format(cname), len(class_sets[cname]))

def get_sample_weights(dataset_urls, n_class):
    def softmax(x):
        row_max = x.max()
        row_max = row_max.reshape(-1, 1)
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        s = x_exp / x_sum
        return s

    label_files = []
    for url in dataset_urls:
        _dir = os.path.join(url, 'labels')
        label_files.extend(sorted([os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.tif')]))

    print("> Apply resampling, analysis of the label：")
    class_count = np.zeros(n_class, dtype=np.float64)
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        class_count += _count

    base_prob = np.sum(class_count) / class_count
    base_prob = np.log(base_prob)
    base_prob = softmax(base_prob)

    sampling_weights = []
    for label_path in tqdm(label_files):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        _count = np.bincount(label.flatten(), minlength=n_class)
        _prob = _count * base_prob
        _prob = np.sum(_prob) / label.size
        sampling_weights.append(_prob)

    return np.array(sampling_weights, dtype=np.float32)

if __name__ == '__main__':
    src_dir = './DATASET/v4/256/train/labels'


