# -*- encoding: utf-8 -*-

import os
import cv2
import numpy as np
import math
import time


def apply_colormap(image, heatmap, alpha=0.5, beta=0.5):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, alpha, heatmap, beta, 0)


def randering_mask(image, mask, n_label, colors, alpha=0.5, beta=0.5):
    colors = np.array(colors)
    mh, mw = mask.shape
    mask = np.eye(n_label)[mask.reshape(-1)]
    mask = np.matmul(mask, colors)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)


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

