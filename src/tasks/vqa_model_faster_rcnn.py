# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from types import SimpleNamespace

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        # from https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
        # https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml

        self.args = SimpleNamespace(
                                # model_file= 'data/faster-rcnn-r101.pth',
                                model_file='data/R-50-FPN.pth',
                                config_file='data/R-50-FPN.yaml',
                                # config_file='../vqa-faster-rcnn/configs/visual_genome_vqa/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512_vqa_test.yaml',
                                batch_size=args.batch_size,
                                num_features=36,
                                feature_name="fc6",
                                confidence_threshold=0,
                                background=True,
                                partition=0)
        self.detection_model = self._build_detection_model()

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        return model

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


