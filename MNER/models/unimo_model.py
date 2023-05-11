import sys

import torch

sys.path.append("..")

from torchcrf import CRF
from torch import nn
from .modeling_unimo import UnimoModel
from transformers.modeling_outputs import TokenClassifierOutput

class UnimoCRFModel(nn.Module):
    def __init__(self, label_list, args, vision_config, text_config):
        super(UnimoCRFModel, self).__init__()
        self.args = args
        print(vision_config)
        print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        self.num_labels  = len(label_list) + 1  # pad
        self.model = UnimoModel(vision_config, text_config, n_class=self.num_labels)

        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.batch_id = 0

    def _cal_coeff(self, vision_seq_in_text, text_seq, labels):
        # vision_seq_in_text: [bs, len1, hidden]
        # text_seq: [bs, len1, hidden]
        # labels: [bs, len1]
        criterion = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=-1)
        tanh = nn.Tanh()
        relu = nn.ReLU(inplace=True)
        alpha = self.args.alpha

        out_v = (
            torch.matmul(
                vision_seq_in_text, # [bs, len1, hidden]
                torch.transpose(self.model.cat_classifier.dense.weight[:, 0:self.vision_config.hidden_size], 0, 1)
                # [labels, hidden] -> [hidden,label]
            ) + # [bs, len1, n_label]
            self.model.cat_classifier.dense.bias / 2
        )

        out_t = (
            torch.matmul(
                text_seq,
                torch.transpose(self.model.cat_classifier.dense.weight[:, self.vision_config.hidden_size:],0,1)
            ) + self.model.cat_classifier.dense.bias / 2
        )   # [bs, len1, n_label]

        score_text = 0
        score_vision = 0
        for b in range(out_t.size(0)):  # [bs, len1, n_label]
            score_vision += sum([
                softmax(out_v[b])[i][labels[b].view(-1)[i]] for i in range(out_t.size(1))
            ])
            score_text += sum([
                softmax(out_t[b])[i][labels[b].view(-1)[i]] for i in range(out_t.size(1))
            ])

        ratio_vision = score_vision / score_text
        ratio_text = 1 / ratio_vision

        if ratio_vision > 1:
            coeff_v = 1 - tanh(alpha * relu(ratio_vision))
            coeff_t = 1
        else:
            coeff_t = 1 - tanh(alpha * relu(ratio_text))
            coeff_v = 1
        return coeff_t, coeff_v


    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        bsz = input_ids.size(0)

        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        # --------------------------------------------------------------------------------
        if len(output) == 3:  # cat
            text_seq, vision_seq_in_text, emissions = output[0], output[1], output[2]
        else:
            assert False
        # --------------------------------------------------------------------------------

        # sequence_output = output.last_hidden_state       # bsz, len, hidden
        # sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        # emissions = self.fc(sequence_output)             # bsz, len, labels
        coeff_t, coeff_v = self._cal_coeff(
            vision_seq_in_text=vision_seq_in_text, text_seq=text_seq, labels=labels
        )
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')  #
        return loss, logits, coeff_v, coeff_t
