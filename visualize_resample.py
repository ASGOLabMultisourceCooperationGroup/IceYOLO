import cv2
import numpy as np
import os
import tifffile
from glob import glob
import rs_utils  # 确保你有这个模块

def resample_segments_yolo(segments, n=1000):
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, dim]) for dim in range(2)], dtype=np.float32).reshape(2, -1).T
        )
    return segments

def resample_segments_custom(segments, n=1000):
    segments = rs_utils.resample_segments([arr.flatten().tolist() for arr in segments], n)
    segments = np.stack([np.around(s, decimals=8).reshape(-1, 2) for s in segments], axis=0)
    return segments

def load_tif_image_rgb(path):
    img = tifffile.imread(path)
    img = img[:, :, :3]  # 取前三通道
    img_min = img.min()
    img_max = img.max()
    if img_max > 255 or img_min < 0:
        img = (img - img_min) / (img_max - img_min) * 255.0
    img = img.clip(0, 255).astype(np.uint8)
    return img

def create_mask(segments, shape):
    """ 根据标签的多边形坐标生成掩码 """
    mask = np.zeros(shape, dtype=np.uint8)
    for points in segments:
        points_int = points.astype(np.int32)
        cv2.fillPoly(mask, [points_int], 255)  # 使用255填充多边形区域
    return mask

def calculate_mask_difference(mask1, mask2):
    """ 计算两个掩码之间的差异像素数量 """
    diff = cv2.absdiff(mask1, mask2)
    diff_pixels = np.sum(diff == 255)  # 统计不同区域的像素数量
    return diff_pixels

def save_image(image, save_path):
    """ 保存图像到指定路径 """
    cv2.imwrite(save_path, image)

def visualize_compare(image_path, label_path, segment_resamples=1000, diff_threshold=1000, output_dir="alignment_vis"):
    # 读图
    image = load_tif_image_rgb(image_path)
    h, w, _ = image.shape

    # 读标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"Warning: {label_path} is empty, skipping this file.")
        return True

    segments = []
    class_ids = []

    # 解析标签
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * w
            y = coords[i+1] * h
            points.append([x, y])
        segments.append(np.array(points, dtype=np.float32))
        class_ids.append(class_id)

    # 重采样
    segments_yolo = resample_segments_yolo([s.copy() for s in segments], n=segment_resamples)
    segments_custom = resample_segments_custom([s.copy() for s in segments], n=segment_resamples)

    # 创建掩码
    mask_yolo = create_mask(segments_yolo, (h, w))
    mask_custom = create_mask(segments_custom, (h, w))

    # 计算掩码差异
    diff_pixels = calculate_mask_difference(mask_yolo, mask_custom)

    # 只有当差异像素大于设定的阈值才展示
    if diff_pixels >= diff_threshold:
        # 随机配色（同一个class同一个颜色）
        np.random.seed(42)
        class_colors = {cid: np.random.randint(0, 256, size=3).tolist() for cid in set(class_ids)}

        # 创建YOLO版本
        image_yolo = image.copy()
        for points, class_id in zip(segments_yolo, class_ids):
            points_int = points.astype(np.int32)
            color = class_colors[class_id]
            cv2.fillPoly(image_yolo, [points_int], color=color)
            cv2.polylines(image_yolo, [points_int], isClosed=True, color=(0, 0, 0), thickness=1)

        # 创建Custom版本
        image_custom = image.copy()
        for points, class_id in zip(segments_custom, class_ids):
            points_int = points.astype(np.int32)
            color = class_colors[class_id]
            cv2.fillPoly(image_custom, [points_int], color=color)
            cv2.polylines(image_custom, [points_int], isClosed=True, color=(0, 0, 0), thickness=1)

        # 保存YOLO版本
        yolo_dir = os.path.join(output_dir, "yolo")
        os.makedirs(yolo_dir, exist_ok=True)
        yolo_filename = os.path.splitext(os.path.basename(image_path))[0] + '_yolo.png'
        yolo_save_path = os.path.join(yolo_dir, yolo_filename)
        save_image(image_yolo, yolo_save_path)
        print(f"Saved YOLO result to {yolo_save_path}")

        # 保存Custom版本
        dynamic_dir = os.path.join(output_dir, "dynamic")
        os.makedirs(dynamic_dir, exist_ok=True)
        dynamic_filename = os.path.splitext(os.path.basename(image_path))[0] + '_dynamic.png'
        dynamic_save_path = os.path.join(dynamic_dir, dynamic_filename)
        save_image(image_custom, dynamic_save_path)
        print(f"Saved custom result to {dynamic_save_path}")

    else:
        print(f"No significant difference detected for {image_path}, skipping visualization.")
        return True

    return True

# ========== 主程序 ==========

images_dir = r"F:\NWPU_YRCC_GFICE\images"
labels_dir = r"F:\NWPU_YRCC_GFICE\labels"

label_files = sorted(glob(os.path.join(labels_dir, '*.txt')))

for label_file in label_files:
    filename = os.path.splitext(os.path.basename(label_file))[0]
    image_file = os.path.join(images_dir, filename + '.tif')

    if not os.path.exists(image_file):
        print(f"Warning: {image_file} not found, skipping...")
        continue

    print(f"Processing {filename}...")

    success = visualize_compare(image_file, label_file)
    if not success:
        print("Exiting visualization...")
        break

cv2.destroyAllWindows()
