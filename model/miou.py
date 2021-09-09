import torch


def nms_multi_category(boxes, threshold=0.7):
    """
    非极大值抑制
    :param boxes: 如果是已经排序好的的, 输入 (x0, y0, x1, y1, cls) 否则输入 (x0, y0, x1, y1, cls, confidence)
    :param threshold: 阈值, 重叠部分大于阈值将会被忽略
    :return: nms 后的框
    """
    if not boxes:
        return []

    if len(boxes[0]) == 6:  # With Confidence
        boxes = sorted(boxes, key=lambda x: -x[-1])
        boxes = list(map(lambda x: x[:5], boxes))

    keep = []
    for x01, y01, x02, y02, current_cls_id in boxes:
        is_keep = True
        for x11, y11, x12, y12, keep_cls_id in keep:
            if keep_cls_id != current_cls_id:
                continue
            x = min(x02, x12) - max(x01, x11)
            y = min(y02, y12) - max(y01, y11)
            if x < 0 or y < 0:
                continue
            if x * y >= (y02 - y01) * (x02 - x01) * threshold:
                is_keep = False
        if is_keep:
            keep.append((x01, y01, x02, y02, current_cls_id))
    return keep


def centernet_to_standard(predict_hm, predict_regs_wh, threshold: float = 0.3):
    result = torch.where(predict_hm > threshold)
    points = list(zip(*result))
    boxes = []
    for (cls_id, y, x) in points:
        [bias_x, bias_y, w, h] = predict_regs_wh[:, int(y), int(x)]
        center_x, center_y, w, h = [i for i in [x + bias_x, y + bias_y, w, h]]
        boxes.append([float(i) for i in [
            center_x - w // 2,
            center_y - h // 2,
            center_x + w // 2,
            center_y + h // 2,
            cls_id,
            predict_hm[cls_id, y, x]
        ]])
    boxes = nms_multi_category(boxes, 0.3)
    return boxes


def iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def mean_iou(outputs, targets):
    selected = []
    ious = []

    for target_box in targets:
        temp = (0, None)
        for output_box in outputs:
            if output_box in selected or output_box[4] != target_box[4]:
                continue
            c_iou = iou(target_box[:4], output_box[:4])
            if c_iou > temp[0]:
                temp = (c_iou, output_box)
        selected.append(temp[1])
        ious.append(temp[0])

    if not ious:
        return 0

    return sum(ious) / len(ious)
