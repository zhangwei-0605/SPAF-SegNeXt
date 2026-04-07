import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import rasterio
from rasterio.transform import Affine

Sugarcane_2_classes = {
    'background': [0, 0, 0],
    'sugarcane': [255, 255, 255]
}

PALETTE = list(Sugarcane_2_classes.values())


def mask_to_onehot(mask, palette):
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.uint8)
    return semantic_map


def read_multiband_image(img_path):
    with rasterio.open(img_path) as src:
        image = src.read()
        transform = src.transform
        crs = src.crs
        meta = src.meta.copy()
        image = np.transpose(image, (1, 2, 0))
    return image, transform, crs, meta


def split_single_image(img_path, target_dir, split_size, overlap, val_rate):
    input_dir, img_name = os.path.split(img_path)
    img_id = img_name.rstrip('_image.tif')
    print(f'> Processing: {img_path}')

    image, img_transform, img_crs, img_meta = read_multiband_image(img_path)

    label_path = os.path.join(input_dir.replace('image', 'label'), img_id + '_label.tif')

    bgr_label = cv2.imread(label_path)
    rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
    onehot = mask_to_onehot(rgb_label, PALETTE)
    mask = np.argmax(onehot, axis=2).astype(np.uint8)

    img_h, img_w, _ = image.shape

    stride = split_size - overlap
    n_h = int(np.ceil((img_h - split_size) / stride)) + 1
    n_w = int(np.ceil((img_w - split_size) / stride)) + 1

    boxes_y1y2x1x2 = []
    for i in range(n_h):
        dh = min(i * stride, img_h - split_size)
        for j in range(n_w):
            dw = min(j * stride, img_w - split_size)
            boxes_y1y2x1x2.append([dh, dh + split_size, dw, dw + split_size])

    total_num = len(boxes_y1y2x1x2)
    val_num = int(total_num * val_rate)
    train_num = total_num - val_num

    np.random.seed(10101)
    np.random.shuffle(boxes_y1y2x1x2)
    train_set = boxes_y1y2x1x2[:train_num]
    val_set = boxes_y1y2x1x2[train_num:]

    for box in train_set + val_set:
        y1, y2, x1, x2 = box
        img_crop = image[y1:y2, x1:x2, :]
        label_crop = mask[y1:y2, x1:x2]
        fname = f"{img_id}_{y1}_{x1}.tif"

        new_transform = Affine(
            img_transform.a,
            img_transform.b,
            img_transform.c + x1 * img_transform.a,
            img_transform.d,
            img_transform.e,
            img_transform.f + y1 * img_transform.e
        )

        is_train = box in train_set
        subset = 'train' if is_train else 'val'

        with rasterio.open(
                os.path.join(target_dir, subset, 'images', fname),
                'w',
                driver='GTiff',
                height=img_crop.shape[0],
                width=img_crop.shape[1],
                count=img_crop.shape[2],
                dtype=img_crop.dtype,
                crs=img_crs,
                transform=new_transform
        ) as dst:
            dst.write(np.transpose(img_crop, (2, 0, 1)))

        cv2.imwrite(os.path.join(target_dir, subset, 'labels', fname), label_crop)



if __name__ == '__main__':
    src_dir = '/root/autodl-tmp/Project/rs-segment-pytorch-main/data'
    dst_dir = '/root/autodl-tmp/Project/rs-segment-pytorch-main/dataset_root/Sugarcane_2400'
    split_size = 512
    overlap = 0
    val_rate = 0.1

    for subset in ['train', 'val']:
        os.makedirs(os.path.join(dst_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, subset, 'labels'), exist_ok=True)

    imgs = sorted([
        osp.join(src_dir, 'image','HuiZong', f)
        for f in os.listdir(osp.join(src_dir, 'image','HuiZong'))
        if f.endswith('.tif')
    ])

    for img_path in tqdm(imgs):
        split_single_image(img_path, dst_dir, split_size, overlap, val_rate)


        
    



