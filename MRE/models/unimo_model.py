import sys

from PIL import Image
sys.path.append("..")

import os
import torch
from torch import nn

import torch.nn.functional as F
from .modeling_unimo import UnimoModel

class UnimoREModel(nn.Module):
    id_map = {
        0: [0, 4],
        1: [4, 8],
        2: [8, 12],
        3: [12, 16],
        4: [16, 20]
    }
    none_id = 20
    RATIO_VISION = 0
    def __init__(self, num_labels, tokenizer, args, vision_config, text_config, clip_model_dict, bert_model_dict):
        super(UnimoREModel, self).__init__()
        self.args = args
        print(vision_config)
        print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        # for re
        vision_config.device = args.device
        self.model = UnimoModel(vision_config, text_config)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            # print(name)
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        # print(vision_names)
        # print(clip_model_dict)
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        print((len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict)))
        self.model.load_state_dict(model_dict)

        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
            task_id=None
    ):
        bsz = input_ids.size(0)

        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        # Author: --------------------------------------------------------------------------------------------------------
        if len(output)==4:
            text_seq, vision_seq, text_logits, vision_logits = output[0], output[1], output[2], output[3]
        elif len(output)==3:
            text_seq, vision_seq, cat_logits = output[0], output[1], output[2]
        else:
            assert False
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=-1)
        tanh = nn.Tanh()
        relu = nn.ReLU(inplace=True)
        alpha = self.args.alpha
        vision_weight = 0.5
        weight_size = self.model.cat_classifier.dense.weight.size(1)    # 其实就是拼接的向量长度
        vision_cls = vision_seq[:,0]    # [bs,dim]
        text_cls = text_seq[:,0]        # [bs,dim]
        out_v = (
            torch.mm(
                vision_cls,
                torch.transpose(self.model.cat_classifier.dense.weight[:, 0:self.vision_config.hidden_size],0,1))
                + self.model.cat_classifier.dense.bias / 2
        )

        out_t = (
            torch.mm(
                text_cls,
                torch.transpose(self.model.cat_classifier.dense.weight[:,self.vision_config.hidden_size:],0,1)
                + self.model.cat_classifier.dense.bias / 2
            )
        )       # [bs, n]
        score_text = sum([softmax(out_t)[i][labels.view(-1)[i]] for i in range(out_t.size(0))])
        score_vision = sum([softmax(out_v)[i][labels.view(-1)[i]] for i in range(out_v.size(0))])
        ratio_vision = score_vision / score_text
        ratio_text = 1 / ratio_vision
        UnimoREModel.RATIO_VISION = ratio_vision.item()    # 暴露给外面，容易获取
        if ratio_vision > 1:
            coeff_v = 1 - tanh(alpha*relu(ratio_vision))
            coeff_t = 1
        else:
            coeff_t = 1 - tanh(alpha*relu(ratio_text))
            coeff_v = 1


        MODE = ['entity', 'cls-sep', 'cat'][2]

        if MODE=='cls-sep':
            loss_text = criterion(text_logits, labels.view(-1))
            loss_vision = criterion(vision_logits, labels.view(-1))
            loss = loss_text + vision_weight * loss_vision
        elif MODE=='entity':
            bsz, seq_len, hidden_size = text_seq.shape
            entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
            for i in range(bsz):
                head_idx = input_ids[i].eq(self.head_start).nonzero().item()
                tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
                head_hidden = text_seq[i, head_idx, :].squeeze()
                tail_hidden = text_seq[i, tail_idx, :].squeeze()
                entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
            entity_hidden_state = entity_hidden_state.to(self.args.device)
            text_logits = self.classifier(entity_hidden_state)
            loss_fn = nn.CrossEntropyLoss()
            loss_text = loss_fn(text_logits, labels.view(-1))
            loss_vision = criterion(vision_logits, labels.view(-1))
            # print(loss_text, "+", loss_vision, "*", vision_weight)
            loss = loss_text + vision_weight * loss_vision
            # print(loss)
        elif MODE=='cat':
            if task_id!=None:
                text_logits = cat_logits.clone()
                offset = UnimoREModel.id_map[task_id]
                cat_logits[:,0:int(offset[0])].data.fill_(-10e10)
                cat_logits[:,int(offset[1]):20].data.fill_(-10e10)
                labels = labels.view(-1)
                labels[torch.where(labels==UnimoREModel.none_id)] = offset[1]
                loss = criterion(
                    torch.cat((cat_logits[:, offset[0]:offset[1]], cat_logits[:, -1].unsqueeze(1)), dim=1),
                    labels-offset[0]
                )
            else:
                # -------------------------------------------------------
                loss = criterion(cat_logits, labels.view(-1))
                text_logits = cat_logits
        else:
            assert False

        if labels is not None:
            return loss, text_logits, coeff_v, coeff_t

        # if labels is not None:
        #     return loss, text_logits, coeff_v, coeff_t
        return text_logits

        # last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        # bsz, seq_len, hidden_size = last_hidden_state.shape
        # entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        # for i in range(bsz):
        #     head_idx = input_ids[i].eq(self.head_start).nonzero().item()
        #     tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
        #     head_hidden = last_hidden_state[i, head_idx, :].squeeze()
        #     tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
        #     entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        # entity_hidden_state = entity_hidden_state.to(self.args.device)
        # logits = self.classifier(entity_hidden_state)
        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss()
        #     return loss_fn(logits, labels.view(-1)), logits
        # return logits
        # --------------------------------------------------------------------------------------------------------------
