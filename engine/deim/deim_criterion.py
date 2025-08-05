#LOSS FUNCTION
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from engine.deim.box_operations import box_center_to_corners, box_iou, generalized_box_iou


class DEIMCriterion(nn.Module):
    def _innit(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, num_classes=80, reg_max=32, boxes_weight_format=None, share_matched_indices=False, mal_alpha=None, use_uni_set=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self.get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, alpha=self.alpha, gamma=self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        idx = self.get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_center_to_corners(src_boxes), box_center_to_corners(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        idx = self.get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_center_to_corners(src_boxes), box_center_to_corners(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)

        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1-target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        idx = self.get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = torch.diag(generalized_box_iou(box_center_to_corners(src_boxes), box_center_to_corners(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else boxes_weight * loss_giou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        losses = {}
        if 'pred_corners' in outputs:
            idx = self.get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                    self.fgl_targets_dn = bbox2distance(ref_points, box_center_to_corners(target_boxes), self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'in_dn' not in outputs:
                    self.fgl_targets = bbox2distance(ref_points, box_center_to_corners(target_boxes), self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(box_center_to_corners(outputs['pred_boxes'][idx]), box_center_to_corners(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['target_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.device)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                                                                          (F.log_softmax(pred_corners / T, dim=1),
                                                                           F.softmax(target_corners.detach() / T,
                                                                                     dim=1))).sum(-1)
                    if 'is_dn' in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if ~mask.any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (_, src) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_go_indices(self, indices, indices_aux_list):
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}

            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx

            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }

        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        indices = self.matcher(outputs_without_aux, targets)['indices']
        self.clear_cache()

        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']

            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)

            indices_go = self.get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for lass in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'enc_aux_outputs' in outputs:
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t['labels'])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        if 'dn_outputs' in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:  # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self.get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_center_to_corners(src_boxes), box_center_to_corners(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(box_center_to_corners(src_boxes), box_center_to_corners(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta['dn_positive_idx'], dn_meta['dn_num_group']
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, device=device, dtype=torch.int64)
                gt_idx = gt_idx.title(dn_num_group)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def unimodal_distribution_focus_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list












































