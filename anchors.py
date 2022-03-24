import numpy as np
import torch
import torch.nn as nn

class Anchors(nn.Module):
    def __init__(self, audio_length=12.8, sr=22050, num_anchors=1280, bidirectional_offset=0.0025):
        super(Anchors, self).__init__()
        self.audio_length = audio_length
        self.sr = sr
        self.num_anchors = num_anchors
        self.bidirectional_offset = bidirectional_offset

    def forward(self, input):
        all_anchors = np.zeros((0, 2)).astype(np.float32)

        for offset_index in range(2):
            anchors = np.zeros((0, 2)).astype(np.float32)
            offset_multiplier = offset_index == 0 and 1 or -1
            for anchor_index in range(self.num_anchors):
                offset = offset_multiplier*self.bidirectional_offset
                position_left = get_anchor_position(self.audio_length, self.sr, anchor_index, self.num_anchors, offset)
                position_right = get_anchor_position(self.audio_length, self.sr, anchor_index + 1, self.num_anchors, offset)

                anchor = np.array([position_left, position_right])
                anchors = np.append(anchors, [anchor], axis=0)

            all_anchors = np.append(all_anchors, anchors, axis=0)

        all_anchors = np.expand_dims(anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def get_anchor_position(audio_length, sr, anchor_index, num_anchors, offset):
    return int((audio_length*anchor_index/num_anchors + offset)*sr)
