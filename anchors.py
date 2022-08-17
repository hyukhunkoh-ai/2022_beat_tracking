import numpy as np
import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self, audio_length=12.8, sr=22050, num_anchors=1280):
        super(Anchors, self).__init__()
        self.audio_length = audio_length
        self.sr = sr
        self.num_anchors = num_anchors

    def forward(self, input):
        anchors = np.zeros((0, 2)).astype(np.float32)
        for anchor_index in range(self.num_anchors):
            position_left = get_anchor_position(self.audio_length, self.sr, anchor_index, self.num_anchors)
            position_right = get_anchor_position(self.audio_length, self.sr, anchor_index + 1, self.num_anchors)

            anchor = np.array([position_left, position_right])
            anchors = np.append(anchors, [anchor], axis=0)

        anchors = np.expand_dims(anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(anchors.astype(np.float32))

def get_anchor_position(audio_length, sr, anchor_index, num_anchors, offset=0):
    return int((audio_length*anchor_index/num_anchors + offset)*sr)
