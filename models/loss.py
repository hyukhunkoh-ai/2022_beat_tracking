import numpy as np
import torch
import torch.nn as nn

def get_activation(act_type,ch=None):
    """ Helper function to construct activation functions by a string.
    Args:
        act_type (str): One of 'ReLU', 'PReLU', 'SELU', 'ELU'.
        ch (int, optional): Number of channels to use for PReLU.

    Returns:
        torch.nn.Module activation function.
    """

    if act_type == "PReLU":
        return nn.PReLU(ch)
    elif act_type == "ReLU":
        return nn.ReLU()
    elif act_type == "SELU":
        return nn.SELU()
    elif act_type == "ELU":
        return nn.ELU()

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 1)

        batch_size, length, channels = out1.shape

        out2 = out1.view(batch_size, length, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=1, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * 1, kernel_size=3, padding=1)

    def forward(self, x):
        print("regression forward", x.shape)
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 1)

        return out.contiguous().view(out.shape[0], -1, 1)

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

            beat_lines = bline_annotation[:, :]
            beat_timepoints = (beat_lines[:, 0] + beat_lines[:, 1])/2
            positive_indices = torch.floor(beat_timepoints*100).long()
            
            anchor_beats_start = positive_indices/100

            ##########################
            # compute the loss for classification
            # 1280, 2
            targets = torch.zeros(classification.shape)

            if torch.cuda.is_available():
                targets = targets.cuda()

            num_positive_anchors = len(positive_indices)

            targets[positive_indices, beat_lines[:, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = (torch.ones(targets.shape) * alpha).cuda()
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            classification_losses.append(cls_loss.sum()/num_positive_anchors)

            ##########################
            # compute the loss for regression

            anchor_widths_pi = 0.01 # anchor_beats_end - anchor_beats_start
            anchor_ctr_x_pi = anchor_beats_start + 0.005 # anchor_beats_start + 0.5*anchor_widths_pi

            gt_widths  = 0.01 # beat_lines[:, 1] - beat_lines[:, 0]
            gt_ctr_x   = beat_lines[:, 0] + 0.005 # beat_lines[:, 0] + 0.5*gt_widths
            # gt_start = beat_lines[:, 0]
            # gt_end = beat_lines[:, 1]

            # clip widths to 1
            #gt_widths  = torch.clamp(gt_widths, min=0.01)

            targets_dx = ((gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi).cuda()
            # targets_dw = torch.log(gt_widths / anchor_widths_pi)
            # targets_distance_start = anchor_beats_start - gt_start
            # targets_distance_end = anchor_beats_end - gt_end

            # targets = torch.stack((targets_distance_start, targets_distance_end))
            # targets = targets.t()

            targets_dx = targets_dx.unsqueeze(1)
            regression_diff = torch.abs(targets_dx - regression[positive_indices, :])

            # 9.0 삭제됨. num_box로 추측했고, 명시된 근거가 없음
            regression_loss = torch.where(
                torch.le(regression_diff, 1.0),
                0.5 * torch.pow(regression_diff, 2),
                regression_diff - 0.5
            )
            regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
