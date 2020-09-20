# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.ops.boxes import batched_nms

from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm
from lxrt.modeling import GeLU
from param import args
from param import timed
from param import timings
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_data import VQADataset
from tasks.vqa_data import VQAEvaluator
from utils.datasets import letterbox
from utils.general import xywh2xyxy

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
            topk = int(os.environ.get('IMG_NUM', 13))
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
        img_path = os.path.join('data', 'train-1k-img', img_id + '.jpg')

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
    dset = VQADataset('train-1k')
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


def get_scale_grid_index(keep_idx, x_shape, x_numel):
    i = 0
    for dim in x_numel:
        if keep_idx >= dim:
            keep_idx -= dim
            i += 1
        else:
            break
    return i, unravel_index(keep_idx, x_shape[i])[1:]


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
        img = letterbox(img0, new_shape=self.img_size, auto=False)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

    @timed
    def postprocess_feature(self, output, shape_info, img_size,
                            conf_threshold=0.3, iou_threshold=0.3,
                            num_per_scale_features=16,
                            ):
        # input image: [3, 256, 320]
        pred, x, features = output  # pred: [1, 5040,85]
        # x[0]: [1, 3, 32, 40, 85], features[0]: [1, 128, 32, 40]
        # x[1]: [1, 3, 16, 20, 85], features[1]: [1, 256, 16, 20]
        # x[2]: [1, 3, 8, 10, 85], features[2]: [1, 512, 8, 10]
        num_scales = len(features)
        # num_proposals = pred.shape[1]
        num_classes = pred.shape[-1] - 5
        batch_size = pred.shape[0]
        device = pred.device
        fs_shape, feat_shape = shape_info
        """3 steps: 
        - prepare boxes and scores
        - do nms, sort boxes by keep indices
        - take either fixed number of boxes or by some threshold (variable #)
        
        optimization, first batch nms, second, set max 320 candidates,
        third, improve index filtering
        """
        feat_list = [[] for _ in range(num_scales)]
        box_list = [[] for _ in range(num_scales)]
        for i in range(batch_size):
            one_pred = pred[i]
            one_mask = one_pred[..., 4] > conf_threshold  # candidates
            one_mask_idx = torch.nonzero(one_mask, as_tuple=True)[0]
            one_pred_s = one_pred[one_mask]

            # Compute conf
            one_pred_s[:, 5:] *= one_pred_s[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            boxes = xywh2xyxy(one_pred_s[:, :4])  # [5040, 4]
            batch_boxes = boxes.unsqueeze(1).expand(
                -1, num_classes, 4)
            one_boxes = batch_boxes.contiguous().view(-1, 4)
            scores = one_pred_s[:, 5:].reshape(-1)
            mask = scores >= conf_threshold
            mask_idx = torch.nonzero(mask, as_tuple=True)[0]
            boxesf = one_boxes[mask]
            scoresf = scores[mask].contiguous()
            # idxsf = idxs[mask].contiguous()
            cols = torch.arange(num_classes, dtype=torch.long)[None, :].to(device)
            num_proposals = one_pred_s.shape[0]
            label_idx = cols.expand(num_proposals, num_classes).reshape(-1)
            labelsf = label_idx[mask]
            keep = batched_nms(boxesf, scoresf, labelsf, iou_threshold)

            proposal_idx, cls_idx = unravel_index(mask_idx, batch_boxes.shape[:-1])
            proposal_idx = proposal_idx[keep]
            proposal_idx = one_mask_idx[proposal_idx]
            cls_idx = cls_idx[keep]
            # cls_idx = one_mask_idx[cls_idx]
            # cls_idx = cls_idx[keep]
            # print('conf filtered num={}, iou_num={}'.format(
            # boxesf.shape, keep.shape))
            boxes /= img_size  # normalize to 0~1
            num_props = len(proposal_idx)
            ss = fs_shape.unsqueeze(1).expand(-1, num_props)
            idx = (proposal_idx//ss).sum(dim=0)
            # x_shape = torch.index_select(feat_shape, 0, idx.long())
            prop_scale_dims = [unravel_index(proposal_idx[idx == nsi],
                                             feat_shape[nsi])[1:]
                               for nsi in range(num_scales)]
            boxes_scale_idx = [keep[idx == nsi] for nsi in range(num_scales)]
            cls_scale_idx = [cls_idx[idx == nsi] for nsi in range(num_scales)]
            feat = [[features[nsi][i][..., dim1, dim2]
                     for dim1, dim2 in zip(*prop_scale_dims[nsi])]
                    for nsi in range(num_scales)]
            box = [boxesf[boxes_scale_idx[nsi]] for nsi in range(num_scales)]
            # cls = [cls_scale_idx[nsi] for nsi in range(num_scales)]
            for ns in range(num_scales):
                if len(feat[ns]) > 0:
                    feat_ns = torch.stack(feat[ns][:num_per_scale_features], 0)
                    feat_list[ns].append(feat_ns)
                    box_list[ns].append(box[ns][:num_per_scale_features])
        for ns in range(num_scales):
            if len(feat_list[ns]) > 0:
                feat_list[ns] = torch.stack(feat_list[ns])
                box_list[ns] = torch.stack(box_list[ns])
                # print('feat_list size:', feat_list[ns].size())
            else:
                feat_list[ns] = None
                box_list[ns] = None
                # print('feat_list size:', feat_list[ns])
        return feat_list, box_list
    @timed
    def preprocess_image(self, img_paths):
        img_tensor = []
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
            # im_infos.append(img0)
        image_tensor = torch.stack(img_tensor)
        return image_tensor

    @timed
    def run_detection(self, image_tensor):
        output = self.model.detection_model(image_tensor)
        return output

    @timed
    def run_vqa(self, feat_list, box_list, sent):
        logit = self.model(feat_list, box_list, sent)
        score, label = logit.max(1)
        return score, label

    def warmup(self, img_paths, sent, shape_info):
        image_tensor = self.preprocess_image(img_paths)
        image_outputs = self.run_detection(image_tensor)
        # get bbox and features
        feat_list, box_list = self.postprocess_feature(
            image_outputs, shape_info, self.img_size,
            args.conf_threshold,
            args.iou_threshold, args.num_per_scale_features)
        score, label = self.run_vqa(feat_list, box_list, sent)
        return score, label

    def real_run(self, img_paths, sent, shape_info):
        image_tensor = self.preprocess_image(img_paths)
        image_outputs = self.run_detection(image_tensor)
        # get bbox and features
        feat_list, box_list = self.postprocess_feature(
            image_outputs, shape_info, self.img_size,
            args.conf_threshold,
            args.iou_threshold, args.num_per_scale_features)
        score, label = self.run_vqa(feat_list, box_list, sent)
        return score, label

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        self.model.detection_model.eval()
        num_classes = self.model.detection_model.nc
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        import time
        from tqdm import tqdm
        start = 0
        batches = 0
        count = 0
        warmed_up = False
        first_print = True
        device = next(self.model.parameters()).device
        # cols = torch.arange(num_classes, dtype=torch.long)[None, :].to(device)
        feat_shape = [torch.Size([3, self.img_size//i, self.img_size//i])
                      for i in [8, 16, 32]]
        shape_numel = torch.tensor([si.numel() for si in feat_shape]).to(device)
        fs_shape = torch.cumsum(shape_numel, 0)
        feat_shape = torch.tensor(feat_shape).to(device)
        shape_info = (fs_shape, feat_shape)
        with torch.no_grad():
            for i, datum_tuple in tqdm(enumerate(loader)):
                ques_id, img_paths, sent = datum_tuple[:3]
                if batches < 3:
                    print('model warming up {}...'.format(batches))
                    score, label = self.warmup(img_paths, sent, shape_info)
                    warmed_up = True
                else:
                    if warmed_up and first_print:
                        print('model warmed up')
                        time.sleep(3)
                        first_print = False
                        start = time.time()
                    score, label = self.real_run(img_paths, sent, shape_info)
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
        latency = end - start
        per_latency = latency / (batches-3)
        print('pred took {:.4f}s, {:.6f} each, batches={}, count={}'.format(
            latency, per_latency, batches, count))
        print('timings: \n{}'.format(
            '\n'.join(['{}: {:5f}'.format(k, sum(v[3:])/(batches-3))
                       for k, v in timings.items()])))
        # print('details: \n{}'.format('\n'.join(['{}: {}'.format(k, v)
        # for k, v in timings.items()])))
        with open(args.profile_save or 'profile.txt', 'w') as f:
            for k, v in timings.items():
                f.write('{},{}\n'.format(k, ','.join([str(vi) for vi in v])))
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
