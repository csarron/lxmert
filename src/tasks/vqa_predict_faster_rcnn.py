# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import glob

import cv2
import numpy as np
import torch
from PIL import Image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
from types import SimpleNamespace

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQAEvaluator
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
            topk = args.tiny_num
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


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


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
        self.args = self.model.args
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
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

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat = feats[i][keep_boxes]
            feat_list.append(feat)
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = im_infos[i]['height'], im_infos[i]['width']
            # boxes = boxes.copy()
            bbox[:, (0, 2)] /= img_w
            bbox[:, (1, 3)] /= img_h
            info_list.append(bbox)
            # print('size:', bbox.size(), feat.size())

        return feat_list, info_list

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
        print('model set up, starting warming up prediction...')
        count = 0
        batches = 0
        # with torch.no_grad(), profiler.profile(record_shapes=True) as prof:
        with torch.no_grad():
            for i, datum_tuple in tqdm(enumerate(loader)):
                ques_id, img_paths, sent = datum_tuple[:3]   # Avoid seeing ground truth
                img_tensor, im_scales, im_infos = [], [], []
                for img_path in img_paths:
                    im, im_scale, im_info = self._image_transform(img_path)
                    # im, im_scale, im_info = img_item
                    img_tensor.append(im)
                    im_scales.append(im_scale)
                    im_infos.append(im_info)
                current_img_list = to_image_list(img_tensor, size_divisible=32)
                # print('current_img_list.device', current_img_list.tensors.size())
                current_img_list = current_img_list.to("cuda")
                output = self.model.detection_model(current_img_list)

                # get bbox and features
                feat_list, info_list = self._process_feature_extraction(
                    output, im_scales, im_infos, self.args.feature_name,
                    self.args.confidence_threshold,
                    )
                feats = torch.stack(feat_list)
                boxes = torch.stack(info_list)
                # feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                batches += 1
                if batches >=2:
                    break
        batches = 0
        count = 0
        print('model warmed up, starting predicting...')
        with torch.no_grad(), torchprof.Profile(self.model, use_cuda=True) as prof:
            for i, datum_tuple in tqdm(enumerate(loader)):
                ques_id, img_paths, sent = datum_tuple[:3]   # Avoid seeing ground truth
                img_tensor, im_scales, im_infos = [], [], []
                for img_path in img_paths:
                    im, im_scale, im_info = self._image_transform(img_path)
                    # im, im_scale, im_info = img_item
                    img_tensor.append(im)
                    im_scales.append(im_scale)
                    im_infos.append(im_info)
                current_img_list = to_image_list(img_tensor, size_divisible=32)
                # print('current_img_list.device', current_img_list.tensors.size())
                current_img_list = current_img_list.to("cuda")
                output = self.model.detection_model(current_img_list)

                # get bbox and features
                feat_list, info_list = self._process_feature_extraction(
                    output, im_scales, im_infos, self.args.feature_name,
                    self.args.confidence_threshold,
                    )
                feats = torch.stack(feat_list)
                boxes = torch.stack(info_list)
                # feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                batches += 1
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
                    count += 1
        print(prof.display(show_events=False))
        end = time.time()
        trace, event_lists_dict = prof.raw()
        import pickle
        with open(args.profile_save or 'profile.pk', 'wb') as f:
            pickle.dump(event_lists_dict, f)
        print('prediction finished!', end-start, batches, count)
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
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))
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



