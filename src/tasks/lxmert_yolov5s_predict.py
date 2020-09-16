# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import glob
import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

import cv2
import numpy as np
import torch
from PIL import Image

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_data import VQADataset, VQAEvaluator
from utils.datasets import letterbox
from utils.general import scale_coords
from utils.general import xywh2xyxy

TINY_IMG_NUM = 10
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class VQATorchDataset(Dataset):

    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = 10
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Only load topk data
        load_topk = topk
        self.data = self.raw_dataset.data[:load_topk]
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        # img_info = self.imgid2img[img_id]
        # obj_num = img_info['num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # assert obj_num == len(boxes) == len(feats)

        # # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # boxes = boxes.copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(-boxes, 0+1e-5)
        img_path = os.path.join('data', 'val2014', img_id + '.jpg')

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, img_path, ques, target
        else:
            return ques_id, img_path, ques


def get_data_tuple(splits: str, bs: int, shuffle=False,
                   drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


MAX_VQA_LENGTH = 20


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def get_grid_index(keep_idx, x):
    dims = [xi[..., 0].numel() for xi in x]
    i = 0
    for dim in dims:
        if keep_idx >= dim:
            keep_idx -= dim
            i += 1
        else:
            break
    return i, unravel_index(keep_idx, x[i][..., 0].shape)[1:]


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
        device = torch.device('cpu')
        detection_model = torch.load(args.detection_model, map_location=device)
        self.detection_model = detection_model['model'].float().fuse().eval()

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


class VQA:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)
        # self.args = args
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        if torch.cuda.is_available():
            self.model = self.model.half().cuda()
            self.model.detection_model = self.model.detection_model.half().cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.img_size = args.img_size

    def _image_transform(self, path):
        img0 = cv2.imread(path)  # BGR
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

    def postprocess_feature(self, img, im0, output,
                         conf_threshold=0.3, iou_threshold=0.3,
                         num_per_scale_features=8,
                         ):
        # input image: [3, 256, 320]
        pred, x, features = output  # pred: [1, 5040,85]
        # x[0]: [1, 3, 32, 40, 85], features[0]: [1, 128, 32, 40]
        # x[1]: [1, 3, 16, 20, 85], features[1]: [1, 256, 16, 20]
        # x[2]: [1, 3, 8, 10, 85], features[2]: [1, 512, 8, 10]
        num_scales = len(features)
        """3 steps: 
        - prepare boxes and scores
        - do nms, sort boxes by keep indices
        - take either fixed number of boxes or by some threshold (variable #)
        """
        batch_size = pred.shape[0]
        num_classes = pred[0].shape[1] - 5
        feat_list = [[] for _ in range(num_scales)]
        info_list = [[] for _ in range(num_scales)]
        cls_list = [[] for _ in range(num_scales)]
        for i in range(batch_size):
            one_pred = pred[i]
            # Compute conf
            one_pred[:, 5:] *= one_pred[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            boxes = xywh2xyxy(one_pred[:, :4])  # [5040, 4]

            # scores = one_pred[:, 4]
            max_conf = torch.zeros_like(one_pred[:, 4])  # [5040]
            conf_thresh_tensor = torch.full_like(max_conf, conf_threshold)
            start_index = 0
            for cls_ind in range(start_index, num_classes):
                cls_scores = one_pred[:, cls_ind + 5]  # 5040
                keep = torch.ops.torchvision.nms(
                    boxes, cls_scores, iou_threshold)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )
            sorted_scores, sorted_indices = torch.sort(max_conf,
                                                       descending=True)
            # TODO: use >0 to get variable boxes
            # num_boxes = (sorted_scores != 0).sum()
            # print('num_boxes: {}'.format(num_boxes))
            # keep_boxes = sorted_indices[: num_features]
            # print('img[i].shape', img[i].shape)
            boxes = scale_coords(img[i].shape[1:], boxes, im0[i].shape).round()
            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = im0[i].shape[:2]
            # boxes = boxes.copy()
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            # unravel_index to get original grid indices
            feat = [[] for _ in range(num_scales)]
            bboxes = [[] for _ in range(num_scales)]
            classes = [[] for _ in range(num_scales)]
            for keep_idx in sorted_indices:
                one_x = [x[nsi][i] for nsi in range(num_scales)]
                scale_idx, (dim1, dim2) = get_grid_index(keep_idx, one_x)
                if len(feat[scale_idx]) >= num_per_scale_features:
                    continue
                feat_idx = features[scale_idx][i][..., dim1, dim2]
                feat[scale_idx].append(feat_idx)
                bbox = boxes[keep_idx]
                bboxes[scale_idx].append(bbox)
                cls = one_pred[keep_idx][..., 5:].argmax()
                classes[scale_idx].append(cls)
            for ns in range(num_scales):
                # feat[ns] = torch.stack(feat[ns], 0)
                # classes[ns] = torch.stack(classes[ns], 0)
                # bboxes[ns] = torch.stack(bboxes[ns], 0)

                feat_list[ns].append(torch.stack(feat[ns], 0))
                info_list[ns].append(torch.stack(bboxes[ns], 0))
                cls_list[ns].append(torch.stack(classes[ns], 0))
            # print('size:', bbox.size(), feat.size())
        for ns in range(num_scales):
            feat_list[ns] = torch.stack(feat_list[ns])
            info_list[ns] = torch.stack(info_list[ns])
            cls_list[ns] = torch.stack(cls_list[ns])
        return feat_list, info_list

    def preprocess_image(self, img_paths):
        img_tensor, im_infos = [], []
        for img_path in img_paths:
            img, img0 = self._image_transform(img_path)
            img = torch.from_numpy(img).float()
            if torch.cuda.is_available():
                img = img.cuda()
                img = img.half()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)
            img_tensor.append(img)
            im_infos.append(img0)
        image_tensor = torch.stack(img_tensor)
        return image_tensor, im_infos

    def run_detection(self, image_tensor):
        output = self.model.detection_model(image_tensor)
        return output

    def run_vqa(self, feat_list, info_list, sent):
        logit = self.model(feat_list, info_list, sent)
        score, label = logit.max(1)
        return score, label

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        import time
        from tqdm import tqdm
        import torchprof
        # import torch.autograd.profiler as profiler

        start = time.time()
        # print('model set up, starting warming up prediction...')
        # count = 0
        # batches = 0
        # # with torch.no_grad(), profiler.profile(record_shapes=True) as prof:
        # with torch.no_grad():
        #     for i, datum_tuple in tqdm(enumerate(loader)):
        #         # :3 Avoid seeing ground truth
        #         ques_id, img_paths, sent = datum_tuple[:3]
        #         img_tensor, im_scales, im_infos = [], [], []
        #         for img_path in img_paths:
        #             img, img0 = self._image_transform(img_path)
        #             img = torch.from_numpy(img).float()
        #             if torch.cuda.is_available():
        #                 img = img.cuda()
        #                 img = img.half()  # uint8 to fp16/32
        #             img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #             # if img.ndimension() == 3:
        #             #     img = img.unsqueeze(0)
        #             img_tensor.append(img)
        #             im_infos.append(img0)
        #         image_tensor = torch.stack(img_tensor)
        #         # print(image_tensor.shape)
        #         output = self.model.detection_model(image_tensor)
        #
        #         # get bbox and features
        #         feat_list, info_list = self._process_feature(
        #             img_tensor, im_infos, output,
        #             args.conf_threshold,
        #             args.iou_threshold, args.num_per_scale_features)
        #         # feats = torch.stack(feat_list)
        #         # boxes = torch.stack(info_list)
        #         # feats, boxes = feats.cuda(), boxes.cuda()
        #         logit = self.model(feat_list, info_list, sent)
        #         score, label = logit.max(1)
        #         batches += 1
        #         if batches >= 2:
        #             break
        batches = 0
        count = 0
        print('model warmed up, starting predicting...')
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        # torchprof.Profile(self.model, use_cuda=torch.cuda.is_available()) as prof
        with torch.no_grad():
            for i, datum_tuple in tqdm(enumerate(loader)):
                ques_id, img_paths, sent = datum_tuple[:3]
                image_tensor, im_infos = self.preprocess_image(img_paths)
                image_outputs = self.run_detection(image_tensor)
                # get bbox and features
                feat_list, info_list = self.postprocess_feature(
                    image_tensor, im_infos, image_outputs,
                    args.conf_threshold,
                    args.iou_threshold, args.num_per_scale_features)
                score, label = self.run_vqa(feat_list, info_list, sent)
                batches += 1
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
                    count += 1
        # print(prof.display(show_events=False))
        # profiler.stop()
        end = time.time()
        # trace, event_lists_dict = prof.raw()
        # import pickle
        # with open(args.profile_save or 'profile.pk', 'wb') as f:
        #     pickle.dump(event_lists_dict, f)
        print('prediction finished!', end - start, batches, count)
        # with open(args.profile_save or 'profile.html', 'w') as f:
        #     f.write(profiler.output_html())
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        # FIXME: load correct checkpoints
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        print(self.args.model_file)
        checkpoint = torch.load(self.args.model_file,
                                map_location=torch.device("cpu"))
        detection_stat_dict = checkpoint.pop("model")
        state_dict.update(detection_stat_dict)
        # print(checkpoint)
        # load_state_dict(model, checkpoint.pop("model"))
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        # args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=args.batch_size,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
