# -*- encoding: utf-8 -*-

# 导入必要的库
import cv2  # OpenCV库，用于图像处理
import random  # 随机数生成库
import colorsys  # 颜色转换库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示库
from skimage import segmentation, measure, morphology, color  # 图像分割、测量、形态学操作和颜色转换库
from scipy import ndimage as ndi  # SciPy的图像处理模块
from scipy.ndimage.morphology import distance_transform_edt  # 距离变换库



def apply_watershed(prob_map, seed_thr, mask):
    seeds = (prob_map > seed_thr).astype(np.uint8)
    seeds, _ = ndi.label(seeds)

    prob_reverse = (1. - prob_map)
    labels = morphology.watershed(prob_reverse, seeds, mask=mask, watershed_line=True)

    return labels, seeds


def mask_remove_large_objects(mask, area_threshold):
    large_objects = morphology.remove_small_objects(ar=mask, min_size=area_threshold, connectivity=1, in_place=False)
    return mask - large_objects


def mask_remove_small_holds(mask, area_threshold):
    labeled = measure.label(mask, connectivity=1)
    new_mask = morphology.remove_small_holes(ar=labeled, area_threshold=area_threshold, connectivity=1)
    new_mask = (new_mask > 0).astype(np.uint8)
    return new_mask


def mask_remove_small_objects(mask, area_threshold):
    labeled = measure.label(mask, connectivity=1)
    new_mask = morphology.remove_small_objects(ar=labeled, min_size=area_threshold, connectivity=1)
    new_mask = (new_mask > 0).astype(np.uint8)
    return new_mask


def mask_remove_small_objects_multiclasse(masks, area_thresholds):
    assert masks.ndim == 3, print('check input masks,expect 3-dimension array, got {}.'.format(masks.ndim))

    H, W, K = masks.shape
    class_num = K - 1

    if isinstance(area_thresholds, list):
        area_thresholds = np.array(area_thresholds)
    else:
        area_thresholds = np.ones(class_num) * area_thresholds
    assert area_thresholds.size == class_num

    new_masks = np.zeros_like(masks)
    for i in range(class_num):
        new_masks[:, :, i + 1] = mask_remove_small_objects(masks[:, :, i + 1], area_thresholds[i])

    _back_ground = np.zeros((H, W))
    _back_ground[np.sum(new_masks, axis=-1) == 0] = 1
    new_masks[:, :, 0] = _back_ground
    return new_masks


def mask_to_onehot(mask, num_classes):
    H, W = mask.shape
    _onehot = np.eye(num_classes)[mask.reshape(-1)]
    _onehot = _onehot.reshape(H, W, num_classes)
    return _onehot


def img_to_onehot(mask, palette):
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask):
    _mask = np.argmax(mask, axis=-1)
    return _mask


def onehot_to_colormap(mask, palette):
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def mask_to_binary_edges(mask, radius=2):
    if radius < 0:
        return mask

    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dist = distance_transform_edt(mask_pad) + distance_transform_edt(1.0 - mask_pad)
    dist = dist[1:-1, 1:-1]
    dist[dist > radius] = 0
    edgemap = dist
    edgemap = np.expand_dims(edgemap, axis=2)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


def binary_mask_to_polygon(binary_mask, tolerance=0):
    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)

    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def sementic_splash(image, mask, n_label, colors=None, alpha=0.5, beta=0.5):
    if colors is not None:
        colors = np.array(colors)
    else:
        colors = random_colors(n_label)
        colors = np.array(colors) * 255

    mh, mw = mask.shape
    mask = np.eye(n_label)[mask.reshape(-1)]
    mask = np.matmul(mask, colors)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)


def instance_splash(image, masks, onehot=True, colors=None, alpha=0.5):
    if onehot:
        N = masks.shape[-1]
    else:
        N = masks.max()

    colors = colors or random_colors(N)
    masked_image = image.astype(np.uint32).copy()

    for i in tqdm(range(N)):
        color = colors[i]
        if onehot:
            mask = masks[:, :, i]
        else:
            mask = (masks == i + 1).astype(np.uint8)

        for c in range(3):
            masked_image[:, :, c] = np.where(
                mask == 1,
                masked_image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                masked_image[:, :, c]
            )
    return masked_image.astype(np.uint8)


def splash_instances_to_image_cv2(image, mask, colors=None, alpha=0.4):
    insts_map, _ = ndi.label(mask)
    masked_image = color.label2rgb(
        insts_map,
        image=image,
        colors=colors,
        alpha=alpha,
        bg_label=0,
        bg_color=(0, 0, 0),
        image_alpha=1
    )
    masked_image = (masked_image * 255).astype(np.uint8)
    return masked_image

