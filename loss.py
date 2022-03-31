import numpy as np
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bline_annotation = annotations[j, :, :]
            bline_annotation = bline_annotation[bline_annotation[:, 2] != -1] # -1은 padding value
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            beat_lines = bline_annotation[:, :2]
            beat_timepoints = (beat_lines[:, 0] + beat_lines[:, 1])/2
            positive_indices = torch.floor(beat_timepoints*100)
            
            anchor_beats_start = anchor_beats_end/100
            anchor_beats_end = positive_indices + 0.01

            ##########################
            # compute the loss for classification
            # 1280, 2
            targets = torch.zeros(classification.shape)

            if torch.cuda.is_available():
                targets = targets.cuda()

            num_positive_anchors = len(positive_indices)

            targets[positive_indices, beat_lines[positive_indices, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            ##########################
            # compute the loss for regression

            anchor_widths_pi = 0.01 # anchor_beats_end - anchor_beats_start
            anchor_ctr_x_pi = anchor_beats_start + 0.005 # anchor_beats_start + 0.5*anchor_widths_pi

            gt_widths  = 0.01 # beat_lines[:, 1] - beat_lines[:, 0]
            gt_ctr_x   = beat_lines[:, 0] + 0.005 # beat_lines[:, 0] + 0.5*gt_widths
            # gt_start = beat_lines[:, 0]
            # gt_end = beat_lines[:, 1]

            # clip widths to 1
            gt_widths  = torch.clamp(gt_widths, min=0.01)

            targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
            # targets_dw = torch.log(gt_widths / anchor_widths_pi)
            # targets_distance_start = anchor_beats_start - gt_start
            # targets_distance_end = anchor_beats_end - gt_end

            # targets = torch.stack((targets_distance_start, targets_distance_end))
            # targets = targets.t()

            regression_diff = torch.abs(targets - regression[positive_indices, :])

            # 9.0 삭제됨. num_box로 추측했고, 명시된 근거가 없음
            regression_loss = torch.where(
                torch.le(regression_diff, 1.0),
                0.5 * torch.pow(regression_diff, 2),
                regression_diff - 0.5
            )
            regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


