import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

import logging
logger = logging.getLogger(__name__)


class MMPNERBertProcessor(object):
    def __init__(self, data_path, task_num, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor
        # Author -------------------------------------------------------------------
        self.task_num = task_num
        # ------------------------------------------------------------------------

    def load_from_files(self, mode="train"):
        '''
        data_path:{
                'train': 'data/twitter2017/train.txt',
                'dev': 'data/twitter2017/valid.txt',
                'test': 'data/twitter2017/test.txt',
                'predict': 'data/twitter2017/test.txt',
                'train_auximgs': 'data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': 'data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
                'predict_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
                'img2crop': 'data/twitter17_detect/twitter17_img2crop.pth'
        }
        :param mode:
        :return:
        '''
        data = []
        for n in range(self.task_num):
            load_file = self.data_path[mode] + str(n) + '.txt'
            logger.info("Loading data from {}".format(load_file))
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                raw_words, raw_targets = [], []
                raw_word, raw_target = [], []
                imgs = []
                for line in lines:
                    if line.startswith("IMGID:"):
                        img_id = line.strip().split('IMGID:')[1] + '.jpg'
                        imgs.append(img_id)
                        continue
                    if line != "\n":
                        raw_word.append(line.split('\t')[0])
                        label = line.split('\t')[1][:-1]
                        if 'OTHER' in label:
                            label = label[:2] + 'MISC'
                        raw_target.append(label)
                    else:
                        raw_words.append(raw_word)
                        raw_targets.append(raw_target)
                        raw_word, raw_target = [], []
            assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words),
                                                                                        len(raw_targets), len(imgs))
            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)

            rcnn_imgs = torch.load(self.data_path['img2crop'])

            data.append(
                {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs, "rcnn_imgs":rcnn_imgs}
            )
        return data

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs))
        aux_imgs = None
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

        rcnn_imgs = torch.load(self.data_path['img2crop'])

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs, "rcnn_imgs":rcnn_imgs}



class MMPNERBertDataset(Dataset):
    ID = 0
    _BOUNDS = 0
    NONE = None    
    def __init__(self, processor, label_mapping, transform, img_path=None, aux_path=None, max_seq=40, ignore_idx=-100, aux_size=128, rcnn_size=64, mode='train') -> None:
        self.processor = processor
        self.transform = transform
        # Author------------------------------------------------------
        self.data_dict = processor.load_from_files(mode)
        # ----------------------------------------------------------
        self.tokenizer = processor.tokenizer
        self.label_mapping = label_mapping
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_path[mode]  if aux_path is not None else None
        self.rcnn_img_path = 'data'
        self.mode = mode
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        # MIKE------------------------------------------------------
        self._init()
        MMPNERBertDataset._BOUNDS = self.processor.task_num
        # ----------------------------------------------------------

    def _init(self):
        self.data_dict.append({
            "words": [],
            "targets": [],
            "imgs": [],
            "aux_imgs": self.data_dict[0]["aux_imgs"],
            "rcnn_imgs": self.data_dict[0]["rcnn_imgs"]
        })
    
    def __len__(self):
        return len(self.data_dict[MMPNERBertDataset.ID]['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict[MMPNERBertDataset.ID]['words'][idx], self.data_dict[MMPNERBertDataset.ID]['targets'][idx], self.data_dict[MMPNERBertDataset.ID]['imgs'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        # labels = [self.ignore_idx] + labels + [self.ignore_idx]*(self.max_seq-len(labels)-1)
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict[MMPNERBertDataset.ID]['aux_imgs']:
                    aux_img_paths  = self.data_dict[MMPNERBertDataset.ID]['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                for i in range(3-len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    img = img.split('.')[0]
                    if img in self.data_dict[MMPNERBertDataset.ID]['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict[MMPNERBertDataset.ID]['rcnn_imgs'][img]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    for i in range(3-len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size))) 

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs, rcnn_imgs

                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)

    def adding_to_buffer(self, new_data):
        # new_data = [
        #     {"wowrds":, "targets":,},
        #     {"wowrds":, "targets":,},
        # ]

        keys = ["words", "targets", "imgs"]
        for item in new_data:
            for key in keys:
                self.data_dict[self.processor.task_num][key].append(item[key])

    def adding(self, task_id, new_data):
        assert False

    def get_data_for_adding(self, task_id, idx):
        data = {}
        keys = ["words", "targets", "imgs"]
        for key in keys:
            data[key] = self.data_dict[task_id][key][idx]
        return data

    @classmethod
    def update(cls, id):
        assert id<=cls._BOUNDS
        cls.ID = id

