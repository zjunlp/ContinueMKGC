import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)


class MMREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, task_num, clip_processor=None, aux_processor=None, rcnn_processor=None, local_files_only=True):
        self.data_path = data_path
        self.re_path = re_path
        self.task_num = task_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True, local_files_only=local_files_only)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    # Author add
    def load_from_files(self, mode="train"):
        '''
        data_path:{
            'train': 'data/txt/sep2/ours_train_',
            'dev': 'data/txt/sep2/ours_val_',
            'test': 'data/txt/sep2/ours_test_',
            'train_auximgs': 'data/txt/mre_train_dict.pth',     # {data_id : object_crop_img_path}
            'dev_auximgs': 'data/txt/mre_dev_dict.pth',
            'test_auximgs': 'data/txt/mre_test_dict.pth',
            'train_img2crop': 'data/img_detect/train/train_img2crop.pth',
            'dev_img2crop': 'data/img_detect/val/val_img2crop.pth',
            'test_img2crop': 'data/img_detect/test/test_img2crop.pth'
        }
        '''
        # lifelong
        data = []
        n_task = self.task_num    # 分成10组
        for n in range(n_task):
            load_file = self.data_path[mode] + str(n) + '.txt'
            logger.info("Loading data from {}".format(load_file))
            with open(load_file, "r", encoding="utf-8") as f:
                '''
                {
                    'token': ['The', 'latest', 'Arkham', 'Horror', 'LCG', 'deluxe', 'expansion', 'the', 'Circle', 'Undone', 'has', 'been', 'released', ':'], 
                    'h': {
                        'name': 'Circle Undone', 
                        'pos': [8, 10]
                    }, 
                    't': {
                        'name': 'Arkham Horror LCG', 
                        'pos': [2, 5]
                    }, 
                    'img_id': 'twitter_19_31_16_6.jpg', 
                    'relation': '/misc/misc/part_of'
                }

                '''
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                for i, line in enumerate(lines):
                    line = ast.literal_eval(line)  # str to dict
                    words.append(line['token'])
                    relations.append(line['relation'])
                    heads.append(line['h'])  # {name, pos}
                    tails.append(line['t'])
                    imgids.append(line['img_id'])
                    dataid.append(i)
                assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))
            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]  # data/txt/mre_train_dict.pth
            aux_imgs = torch.load(aux_path)
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  # data/img_detect/train/train_img2crop.pth
            data.append(
                {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                    'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}
            )
        return data

    def load_from_file(self, mode="train"):
        '''
        data_path:{
            'train': 'data/txt/ours_train.txt',
            'dev': 'data/txt/ours_val.txt',
            'test': 'data/txt/ours_test.txt',
            'train_auximgs': 'data/txt/mre_train_dict.pth',     # {data_id : object_crop_img_path}
            'dev_auximgs': 'data/txt/mre_dev_dict.pth',
            'test_auximgs': 'data/txt/mre_test_dict.pth',
            'train_img2crop': 'data/img_detect/train/train_img2crop.pth',
            'dev_img2crop': 'data/img_detect/val/val_img2crop.pth',
            'test_img2crop': 'data/img_detect/test/test_img2crop.pth'
        }

        :param mode:
        :return:
        '''
        load_file = self.data_path[mode]                        # data/txt/ours_train.txt
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            '''
            {
                'token': ['The', 'latest', 'Arkham', 'Horror', 'LCG', 'deluxe', 'expansion', 'the', 'Circle', 'Undone', 'has', 'been', 'released', ':'], 
                'h': {
                    'name': 'Circle Undone', 
                    'pos': [8, 10]
                }, 
                't': {
                    'name': 'Arkham Horror LCG', 
                    'pos': [2, 5]
                }, 
                'img_id': 'twitter_19_31_16_6.jpg', 
                'relation': '/misc/misc/part_of'
            }

            '''
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h']) # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        aux_imgs = None
        aux_path = self.data_path[mode+"_auximgs"]              # data/txt/mre_train_dict.pth
        aux_imgs = torch.load(aux_path)
        rcnn_imgs = torch.load(self.data_path[mode+'_img2crop'])    # data/img_detect/train/train_img2crop.pth
        return {'words':words, 'relations':relations, 'heads':heads, 'tails':tails, 'imgids': imgids, 'dataid': dataid, 'aux_imgs':aux_imgs, "rcnn_imgs":rcnn_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key:[] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id

class MMREDataset(Dataset):
    ID = 0
    NONE = None
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64, mode="train") -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode]  if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data'
        self.mode = mode
        self.data_dict = self.processor.load_from_files(mode)    # self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        # self.buffer = {}
        self._init()

    def _init(self):
        # ['words', 'relations', 'heads', 'tails', 'imgids', 'dataid']
        # ID为self.processor.task_num
        self.data_dict.append({
            "words":[],"relations":[],"tails":[],"heads":[],"imgids":[],"dataid":[],
            "aux_imgs":self.data_dict[0]["aux_imgs"],
            "rcnn_imgs":self.data_dict[0]["rcnn_imgs"]
        })
    
    def __len__(self):
        # print("Author:", len(self.data_dict[MMREDataset.ID]['words']))
        return len(self.data_dict[MMREDataset.ID]['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = \
            self.data_dict[MMREDataset.ID]['words'][idx], \
            self.data_dict[MMREDataset.ID]['relations'][idx], \
            self.data_dict[MMREDataset.ID]['heads'][idx], \
            self.data_dict[MMREDataset.ID]['tails'][idx], \
            self.data_dict[MMREDataset.ID]['imgids'][idx]
        item_id = self.data_dict[MMREDataset.ID]['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        # 为头实体和尾实体添加特殊token
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)
        
        re_label = self.re_dict[relation]   # label to id

         # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict[MMREDataset.ID]['aux_imgs']:
                    aux_img_paths  = self.data_dict[MMREDataset.ID]['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                    
                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                for i in range(3-len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict[MMREDataset.ID]['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict[MMREDataset.ID]['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]
                    
                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)
                    
                    # padding
                    for i in range(3-len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size))) 

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, rcnn_imgs

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs
            

        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)

    def adding_to_buffer(self, new_data):
        # new_data = [
        #       {"words": , "relations":, },
        #       {"words": , "relations":, },
        # ]
        keys = ['words', 'relations', 'heads', 'tails', 'imgids', 'dataid']
        for item in new_data:
            for key in keys:
                self.data_dict[self.processor.task_num][key].append(item[key])


    def adding(self, task_id, new_data):
        # new_data = [
        #       {"words": , "relations":, },
        #       {"words": , "relations":, },
        # ]
        keys = ['words', 'relations', 'heads', 'tails', 'imgids', 'dataid']
        for item in new_data:
            for key in keys:
                self.data_dict[task_id][key].append(item[key])
                self.data_dict[self.processor.task_num][key].append(item[key])
                # if key not in self.buffer:
                #     self.buffer[key] = []
                # self.buffer[key].append(item[key])

    def get_data_for_adding(self, task_id, idx):
        data = {}
        keys = ['words','relations','heads','tails','imgids','dataid']
        for key in keys:
            data[key] = self.data_dict[task_id][key][idx]
        return data

    @classmethod
    def update(cls, id):
        cls.ID = id




