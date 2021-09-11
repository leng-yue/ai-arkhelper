import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from definition import TASKS, ACTIONS


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        raise Exception("Image bounding box error")

    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


class ArkDataset(Dataset):
    def __init__(self, folder: Union[Path, str], transform: T = None, down_ratio: int = 4, max_objs: int = 128):
        """
        CenterNet 数据集
        :param folder: 目录
        :param transform: 转换
        :param down_ratio: 降采样率
        :param max_objs: 最多目标数
        """

        if transform is None:
            transform = T.ToTensor()

        self.images = list(Path(folder).rglob("**/*.jpg"))
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item: int):
        try:
            return self._get_item(item)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'Fail to load item {item}: {self.images[item]}, {str(e)}')

    def _get_item(self, item: int):
        path = self.images[item]
        im = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(im, cv2.IMREAD_COLOR)

        # 解析信息
        temp = path.name.replace(".jpg", "").split("_")[:-1]
        action_idx = ACTIONS.index(temp[1])
        task_idx = TASKS.index(temp[0])
        boxes = temp[21:]

        raw_h, raw_w = image.shape[:2]
        image = self.transform(image)  # 1, H, W
        new_h, new_w = image.shape[1:]

        # OneHot 编码操作空间
        actions = np.zeros((len(TASKS), new_h, new_w))
        actions[task_idx] = 1
        image = np.concatenate([image, actions]).astype(np.float32)

        # 计算生成 HeatMap
        output_h = image.shape[1] // self.down_ratio
        output_w = image.shape[2] // self.down_ratio

        hm = np.zeros((1, output_h, output_w), dtype=np.float32)
        regs_wh = np.zeros((4, output_h, output_w), dtype=np.float32)  # bias + width / height
        ind_masks = np.zeros((4, output_h, output_w), dtype=np.float32)

        for box in boxes:
            x0, y0, x1, y1 = list(map(int, box.split(",")))

            x0 = x0 / raw_w * new_w
            y0 = y0 / raw_h * new_h
            x1 = x1 / raw_w * new_w
            y1 = y1 / raw_h * new_h

            w = (x1 - x0 + 1) / self.down_ratio
            h = (y1 - y0 + 1) / self.down_ratio

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            real_cx = (x0 + x1) * 0.5 / self.down_ratio
            real_cy = (y0 + y1) * 0.5 / self.down_ratio
            icx = int(real_cx)
            icy = int(real_cy)

            draw_gaussian(hm[0], (icx, icy), radius)

            range_expand = radius
            for cx in range(icx - range_expand, icx + range_expand + 1):
                for cy in range(icy - range_expand, icy + range_expand + 1):
                    if cy >= output_h or cx >= output_w:
                        continue
                    ind_masks[:, cy, cx] = 1
                    regs_wh[:, cy, cx] = [cx - real_cx, cy - real_cy, w, h]

        return image, action_idx, hm, regs_wh, ind_masks
