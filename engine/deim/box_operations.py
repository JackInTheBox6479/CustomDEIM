import torch
from torchvision.ops.boxes import box_area


def box_center_to_corners(x):
    x_center, y_center, width, height = x.unbind(-1)
    b = [x_center - 0.5 * width.clamp(min=0), y_center - 0.5 * height.clamp(min=0),
         x_center + 0.5 * width.clamp(min=0), y_center + 0.5 * height.clamp(min=0)]
    return torch.stack(b, dim=-1)

def box_corners_to_centers(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - intersection

    iou = intersection / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    #for boxes in [x0, y0, x1, y1] format

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def masks_to_bboxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w  = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

