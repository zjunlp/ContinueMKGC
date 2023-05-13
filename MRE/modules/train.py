import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from models.modeling_unimo_mike import ShareKey, START_SHARE_LAYER, GradKeyAndBias, AttentionReg, BertSelfAttention
from processor.dataset import MMREDataset
from models.unimo_model import UnimoREModel
from torch.utils.data import DataLoader
from torch.nn import functional as F
import gc
sharekey_dict = [
    "config", "v_max_l", "t_max_l", "num_heads", "head_dim", "key", "vision_bias", "text_bias",
    "text_bn", "vision_bn", "layer", "device"
]
attentionreg_dict = [
    "old_model", "old_key", "old_text_bias", "old_vision_bias",
    "attention_text_list", "attention_vision_list", "zero"
]

from .metrics import eval_result

class BertTrainer(object):
    def __init__(self, train_data=None, train_dataset=None, dev_data=None, test_data=None, re_dict=None, model=None, process=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data
        self.train_dataset = train_dataset 
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        self.data_length = []       
        self.new_data = None        
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.n_task = self.args.task_number     
        self.before_multimodal_train()
        self._statistic_data_length()

        self.balance_recorder = {"current":{},"replay":{}}  # {0:[],1:[],2:[],3:[]}
        for i in range(self.n_task):
            self.balance_recorder["current"][i] = []
            self.balance_recorder["replay"][i] = []

        self.fisher = None
        self.optpar = None

    # Author -------------------------------------------------------------------------------------------------------------
    def _statistic_data_length(self):
        for i in range(self.n_task):
            MMREDataset.update(i)
            self.data_length.append(len(self.train_dataset))
        
    def _judge_number(self, number):
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        for n in numbers:
            if number == n:
                return int(n)
        return -1

    def _get_modality_type(self, src_text):
        text = ['text','bert']
        vision = ['clip', 'vision', 'img']
        for t in text:
            if t in src_text:
                return 'text'
        for v in vision:
            if v in src_text:
                return 'vision'
        return 'none'

    def balance(self, coeff_t, coeff_v):
        for name, parms in self.model.named_parameters():
            # layer = str(name).split('.')[1].lower() 
            # print(name, layer)
            layer = name
            # print(layer)
            modality_type = self._get_modality_type(layer)

            if parms.grad == None:
                # print(name)
                continue
            if 'vision' == modality_type:    
                parms.grad = parms.grad * coeff_v + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            elif 'text' == modality_type:   
                parms.grad = parms.grad * coeff_t + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

        # for t in ['vision_bias', 'text_bias']:
        for i in range(START_SHARE_LAYER,ShareKey.layer):
            # print(i)
            parms = ShareKey.text_bias[i]
            parms.grad = parms.grad * coeff_t + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            parms = ShareKey.vision_bias[i]
            parms.grad = parms.grad * coeff_v + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

    def _cal_prgbar_times(self, mode):
        if mode=='train':
            target = self.train_data
            times = self.args.num_epochs
        elif mode == 'test':
            target = self.test_data
            times = 1
        elif mode == 'dev':
            target = self.dev_data
            times = 1
        else:
            assert mode in ['train', 'test', 'dev']
        # 计算次数
        cnt = 0
        for i in range(self.n_task):
            MMREDataset.update(i)
            # print("hello:", len(target))
            cnt += len(target) * times * self.args.batch_size
        MMREDataset.update(0)  
        return cnt

    def get_grad(self):
        grad_key = []
        grad_bias_vision = []
        grad_bias_text = []
        for i in range(GradKeyAndBias.start_layer):
            grad_key.append(0)
            grad_bias_text.append(0)
            grad_bias_vision.append(0)
        for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer+1):
            grad_key.append(ShareKey.key[i].grad.detach().clone())
            grad_bias_text.append(ShareKey.text_bias[i].grad.detach().clone())
            grad_bias_vision.append(ShareKey.vision_bias[i].grad.detach().clone())
        GradKeyAndBias.store_grad(grad_key, grad_bias_text, grad_bias_vision)

    def end_task(self, task_id):
        grad_key = []
        grad_bias_vision = []
        grad_bias_text = []
        cnt = 0 
        for i in range(GradKeyAndBias.start_layer):
            grad_key.append(0)
            grad_bias_text.append(0)
            grad_bias_vision.append(0)
        for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
            grad_key.append(torch.zeros_like(ShareKey.key[i]))
            grad_bias_text.append(torch.zeros_like(ShareKey.text_bias[i]))
            grad_bias_vision.append(torch.zeros_like(ShareKey.vision_bias[i]))

        for batch in self.train_data:
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
            bs = None
            elements = []
            tmp = None
            for b in batch:
                tmp = b
                bs = b.shape[0]
                break
            for element in batch:
                elements.append(element)
            for i in range(bs): 
                cnt += 1
                single_batch = (element[i].unsqueeze(0) for element in elements)
                single_batch = (tmp[i].unsqueeze(0) , *single_batch)
                (loss, logits, coeff_v, coeff_t), labels = self._step(single_batch, mode="train", task_id=task_id)  
                exp_cond_prob = torch.exp(loss.detach().clone())
                loss.backward()
                for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
                    grad_key[i] += exp_cond_prob * ShareKey.key[i].grad.detach().clone() ** 2
                    grad_bias_text[i] += exp_cond_prob * ShareKey.text_bias[i].grad.detach().clone() ** 2
                    grad_bias_vision[i] += exp_cond_prob * ShareKey.vision_bias[i].grad.detach().clone() ** 2
            # ------------------------------------------------------------------

        for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
            grad_key[i] /= cnt
            grad_bias_text[i] /= cnt
            grad_bias_vision[i] /= cnt
        GradKeyAndBias.store_grad(grad_key, grad_bias_text, grad_bias_vision) 
        GradKeyAndBias.save_params()

    def ewc_end_task(self, task_id=None):
        self.fisher = []
        self.optpar = []
        for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
            self.fisher.append(torch.zeros_like(ShareKey.key[i]))
            self.fisher.append(torch.zeros_like(ShareKey.text_bias[i]))
            self.fisher.append(torch.zeros_like(ShareKey.vision_bias[i]))
            self.optpar.append(ShareKey.key[i].data.clone())
            self.optpar.append(ShareKey.text_bias[i].data.clone())
            self.optpar.append(ShareKey.vision_bias[i].data.clone())

        for name, param in self.model.named_parameters():
            if self.args.do_froze:  
                current_name = name.split('.')
                if len(current_name) >= 4:
                    _ = self._judge_number(current_name[3])
                    if _ != -1 and _ >= 0 and _ < self.args.froze_layer:
                        continue    
            if self._judge_use_param(name):     
                self.fisher.append(torch.zeros_like(param))
                self.optpar.append(param.data.clone())

        self.optimizer.zero_grad()
        size = 0                    
        for batch in self.train_data:
            size += 1
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
            (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=task_id)
            loss.backward()
            cnt = 0
            for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
                self.fisher[cnt] += ShareKey.key[i].grad.data.clone().pow(2)
                cnt += 1
                self.fisher[cnt] += ShareKey.text_bias[i].grad.data.clone().pow(2)
                cnt += 1
                self.fisher[cnt] += ShareKey.vision_bias[i].grad.data.clone().pow(2)
                cnt += 1
            for name, param in self.model.named_parameters():
                if self.args.do_froze: 
                    current_name = name.split('.')
                    if len(current_name) >= 4:
                        _ = self._judge_number(current_name[3])
                        if _ != -1 and _ >= 0 and _ < self.args.froze_layer:
                            continue  
                if param.grad != None and self._judge_use_param(name):
                    self.fisher[cnt] += param.grad.data.clone().pow(2)
                    cnt += 1
            assert cnt == len(self.optpar)    
            self.clear()
            AttentionReg.clear_attention_list()
        for i in range(len(self.fisher)):
            self.fisher[i] /= (self.args.batch_size * size)

    def ewc_loss(self):
        idx = 0 
        loss = 0
        for i in range(GradKeyAndBias.start_layer, GradKeyAndBias.end_layer + 1):
            p = ShareKey.key[i]
            l = self.args.gamma * self.fisher[idx]
            l = l * (p - self.optpar[idx]).pow(2)
            loss += l.sum()
            idx += 1

            p = ShareKey.text_bias[i]
            l = self.args.gamma * self.fisher[idx]
            l = l * (p - self.optpar[idx]).pow(2)
            loss += l.sum()
            idx += 1

            p = ShareKey.vision_bias[i]
            l = self.args.gamma * self.fisher[idx]
            l = l * (p - self.optpar[idx]).pow(2)
            loss += l.sum()
            idx += 1

        for name, p in self.model.named_parameters():
            if self._judge_use_param(name):
                l = self.args.gamma * self.fisher[idx]
                l = l * (p - self.optpar[idx]).pow(2)
                loss += l.sum()
                idx += 1
        return loss

    def attention_end_task(self, task_id=None):
        AttentionReg.update_old_model(self.model)
        AttentionReg.update_old_key_and_bias()

    def attention_loss(self, batch, task_id=None, return_logits=False, return_labels=False):
        AttentionReg.clear_attention_list()
        with torch.no_grad():
            AttentionReg.change_ShareKey()  
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs = batch
            AttentionReg.old_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs, task_id=task_id)
            old_vision_attention = AttentionReg.attention_vision_list
            old_text_attention = AttentionReg.attention_text_list
        AttentionReg.change_ShareKey()  
        AttentionReg.clear_attention_list()
        # (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None)
        loss, logits, coeff_v, coeff_t = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs, task_id=task_id)
        new_vision_attention = AttentionReg.attention_vision_list
        new_text_attention = AttentionReg.attention_text_list

        if self.args.type_text.lower() == "both":
            text_loss = AttentionReg.cal_loss(old_text_attention, new_text_attention, merge_type="height") + \
                        AttentionReg.cal_loss(old_text_attention, new_text_attention, merge_type="width")
        elif self.args.type_text.lower() == "none":
            text_loss = 0
        elif self.args.type_text.lower() == "width":
            text_loss = AttentionReg.cal_loss(old_text_attention, new_text_attention, merge_type="width")
        elif self.args.type_text.lower() == "height":
            text_loss = AttentionReg.cal_loss(old_text_attention, new_text_attention, merge_type="height")
        else:
            assert False

        if self.args.type_vision.lower() == "both":
            vision_loss = AttentionReg.cal_loss(old_vision_attention, new_vision_attention, merge_type="width") + \
                          AttentionReg.cal_loss(old_vision_attention, new_vision_attention, merge_type="height")
        elif self.args.type_vision.lower() == "none":
            vision_loss = 0
        elif self.args.type_vision.lower() == "width":
            vision_loss = AttentionReg.cal_loss(old_vision_attention, new_vision_attention, merge_type="width")
        elif self.args.type_vision.lower() == "height":
            vision_loss = AttentionReg.cal_loss(old_vision_attention, new_vision_attention, merge_type="height")
        else:
            assert False

        loss += (self.args.gamma * (text_loss + vision_loss))
        old_vision_attention.clear()
        old_text_attention.clear()

        if return_logits==True and return_labels==False:
            return loss, logits, coeff_v, coeff_t
        elif return_logits==False and return_labels==False:
            return loss, coeff_v, coeff_t
        elif return_logits==True and return_labels==True:
            return loss, logits, coeff_v, coeff_t, labels
        elif return_labels==False and return_logits==True:
            assert False

    def clear(self):
        AttentionReg.attention_text_list.clear()
        AttentionReg.attention_vision_list.clear()

    def save(self,task_id=-1):
        torch.save(self.model.state_dict(), self.args.save_path + f"/model_{task_id}.pth")
        share_parameters = {"ShareKey":{}, "AttentionReg":{}}
        for name in sharekey_dict:
            share_parameters["ShareKey"][name] = eval(f"ShareKey.{name}")
        for name in attentionreg_dict:
            share_parameters["AttentionReg"][name] = eval(f"AttentionReg.{name}")
        torch.save(share_parameters, self.args.save_path +f"/share_{task_id}.pth")
        self.logger.info("Save best model at {}".format(self.args.save_path))

    def load_parameters(self, task_id):
        self.model.load_state_dict(torch.load(self.args.save_path + f"/model_{task_id}.pth"))
        share_parameters = torch.load(self.args.save_path + f"/share_{task_id}.pth")
        print("***** Start Load *****")
        print(f"***** Load from {self.args.save_path}/share_{task_id}.pth, model_{task_id}.pth *****")
        for name in sharekey_dict:
            exec(f"ShareKey.{name} = share_parameters['ShareKey'][name]")
        for name in attentionreg_dict:
            exec(f"AttentionReg.{name}=share_parameters['AttentionReg'][name]")


    def _judge_use_param(self, name):
        if name == "classifier.weight" or name == "classifier.bias" or \
                name == "model.vision_embeddings.position_embedding.weight" or \
                name == "model.vision_post_layernorm.weight" or \
                name == "model.vision_post_layernorm.bias":
            return False
        if "fusion" in name or \
                "9.self_attn.k_proj" in name or \
                "10.self_attn.k_proj" in name or \
                "11.self_attn.k_proj" in name or \
                "9.attention.self.key." in name or \
                "10.attention.self.key." in name or\
                "11.attention.self.key." in name or\
                "text_pooler" in name or \
                "text_classifier.dense" in name or \
                "vision_classifier.dense" in name:
            return False
        return True

    def store_buffer(self, use_attention_loss=False):
        new_data = []
        buffer_size = self.args.buffer_size
        candidates = self.compute_train_score(use_attention_loss=use_attention_loss,
                                              fast_test=self.args.do_random)  # {"index", "influence":, "label"}
        for label in candidates.keys():
            lists = candidates[label]
            lists.sort(key=lambda x: x[list(x.keys())[1]], reverse=True)  # 按照influence从大到小排序
            cnt = 0
            for item in lists:
                cnt += 1
                new_data.append(self.train_dataset.get_data_for_adding(task_id=MMREDataset.ID, idx=item["index"]))
                if cnt >= buffer_size: 
                    break
            # self.train_dataset.adding_to_buffer(new_data=new_data)
            # new_data = []
        self.train_dataset.adding_to_buffer(new_data=new_data)

    def store_for_replay(self, use_attention_loss=False):
        if self.new_data==None:
            self.new_data = []
        buffer_size = self.args.buffer_size
        candidates = self.compute_train_score(use_attention_loss=use_attention_loss, fast_test=self.args.do_random)    # {"index", "influence":, "label"}
        for label in candidates.keys():
            lists = candidates[label]
            lists.sort(key=lambda x:x[list(x.keys())[1]], reverse=True)   # 按照influence从大到小排序
            cnt = 0
            for item in lists:
                cnt += 1
                self.new_data.append(self.train_dataset.get_data_for_adding(task_id=MMREDataset.ID, idx=item["index"]))
                if cnt>=buffer_size:   
                    break
        self.train_dataset.adding(task_id=MMREDataset.ID+1, new_data=self.new_data)

    def compute_train_score(self, use_attention_loss=False, fast_test=False):
        train_dataloader = DataLoader(self.train_dataset, batch_size=2, shuffle=False)
        self.model.eval()
        output_collections = {} # label: [{}, {}, {}]
        pbar = tqdm(total=self.data_length[MMREDataset.ID])
        for idx, inputs in enumerate(train_dataloader):
            pbar.update(1)
            # print(inputs[0][0], self.train_dataset[idx][0])
            if idx>=self.data_length[MMREDataset.ID]:          
                break
            inputs_ = inputs.copy()
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs)
            if use_attention_loss == False:
                (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None) 
            elif use_attention_loss == True:
                loss, logits, coeff_v, coeff_t, labels = self.attention_loss(batch=batch, task_id=None, return_logits=True, return_labels=True)
            if fast_test:    
                outputs = {"index": idx, "influence": torch.rand([1]).item(), "label": labels[0]}
                _label = labels[0].item()
                if _label not in output_collections.keys():
                    output_collections[_label] = []
                output_collections[_label].append(outputs)
                self.clear()
                AttentionReg.clear_attention_list()
                torch.cuda.empty_cache()
                continue
            prob = F.softmax(logits, dim=-1)        # [bs, n]
            prob_gt = torch.gather(prob, 1, labels.unsqueeze(1))

            self.model.zero_grad()
            self.optimizer.zero_grad()

            model_parameters = [param for name, param in self.model.named_parameters() if param.requires_grad and self._judge_use_param(name)]
            for i in range(START_SHARE_LAYER, ShareKey.layer):
                model_parameters.append(ShareKey.key[i])
                model_parameters.append(ShareKey.text_bias[i])
                model_parameters.append(ShareKey.vision_bias[i])
            v = torch.autograd.grad(
                outputs=prob_gt,
                inputs=model_parameters,
                create_graph=False,
            )

            train_dataloader_ = DataLoader(
                self.train_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=True
            )
            s = self.compute_s(
                v=v, train_data_loader=train_dataloader_,
                damp=5e-3, scale=1e4, num_samples=10
            )
            del batch
            gc.collect()
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs_)
            if use_attention_loss == False:
                (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None) 
            elif use_attention_loss == True:
                loss, logits, coeff_v, coeff_t, labels = self.attention_loss(batch=batch, task_id=None, return_logits=True, return_labels=True)
            self.model.zero_grad()
            self.optimizer.zero_grad()

            model_parameters_ = [param for name, param in self.model.named_parameters() if param.requires_grad and self._judge_use_param(name)]
            for i in range(START_SHARE_LAYER, ShareKey.layer):
                model_parameters_.append(ShareKey.key[i])
                model_parameters_.append(ShareKey.text_bias[i])
                model_parameters_.append(ShareKey.vision_bias[i])
            grad_tuple_ = torch.autograd.grad(
                outputs=loss,
                inputs=model_parameters_,
                create_graph=True
            )
            influence = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)]
            influence = sum(influence).item()
            outputs = {"index": idx, "influence": influence, "label": labels[0]}
            _label = labels[0].item()
            if _label not in output_collections.keys():
                output_collections[_label] = []
            output_collections[_label].append(outputs)
            del v,s, grad_tuple_, inputs_, train_dataloader_, influence, model_parameters_, model_parameters, prob, prob_gt, batch
            gc.collect()
            self.clear()
            AttentionReg.clear_attention_list()
            torch.cuda.empty_cache()
        pbar.close()
        self.model.train()
        return output_collections

    def compute_s(self, v, train_data_loader, damp, scale, num_samples, use_attention_loss=False):
        last_estimate = list(v).copy()
        for i in range(num_samples):
            n = 0
            for id, inputs in enumerate(train_data_loader):
                inputs_ = inputs.copy()
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs)
                this_estimate = self.compute_hessian_vector_products(batch=batch, vectors=v, use_attention_loss=use_attention_loss)
                batch_ = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs_)
                for i in batch_:
                    bs = i.shape[0]
                    break
                n += bs
                with torch.no_grad():
                    new_estimate = [
                        a + (1 - damp) * b - c / scale
                        for a, b, c in zip(v, last_estimate, this_estimate)
                    ]
                new_estimate_norm = new_estimate[0].norm().item()
                last_estimate_norm = last_estimate[0].norm().item()
                estimate_norm_diff = new_estimate_norm - last_estimate_norm
                ####
                last_estimate = new_estimate
                if n>num_samples:
                    break
        inverse_hvp = [X / scale for X in last_estimate]
        del this_estimate, inputs_, last_estimate, batch, batch_   
        gc.collect()
        torch.cuda.empty_cache()
        return inverse_hvp

    def compute_hessian_vector_products(self, batch, vectors, use_attention_loss=False):
        if use_attention_loss == False:
            (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None)
        elif use_attention_loss == True:
            loss, coeff_v, coeff_t = self.attention_loss(batch=batch, task_id=None)

        self.model.zero_grad()
        self.optimizer.zero_grad() 

        model_parameters = [param for name, param in self.model.named_parameters() if param.requires_grad and self._judge_use_param(name)]
        for i in range(START_SHARE_LAYER, ShareKey.layer):
            model_parameters.append(ShareKey.key[i])
            model_parameters.append(ShareKey.text_bias[i])
            model_parameters.append(ShareKey.vision_bias[i])

        grad_tuple = torch.autograd.grad(
            outputs=loss,
            inputs=model_parameters,
            create_graph=True,
            allow_unused=True
        )

        model_parameters_ = [param for name, param in self.model.named_parameters() if param.requires_grad and self._judge_use_param(name)]
        for i in range(START_SHARE_LAYER, ShareKey.layer):
            model_parameters_.append(ShareKey.key[i])
            model_parameters_.append(ShareKey.text_bias[i])
            model_parameters_.append(ShareKey.vision_bias[i])

        # cnt = 0
        # for i in grad_tuple:
        #     try:
        #         print(cnt,i.requires_grad)
        #     except:
        #         print("*:",cnt)
        #     cnt+=1
        # cnt = 0
        # for j in vectors:
        #     print(cnt, j.requires_grad)
        #     cnt+=1

        grad_grad_tuple = torch.autograd.grad(
            outputs=grad_tuple,
            inputs=model_parameters_,
            grad_outputs=vectors,
            only_inputs=True,
            allow_unused=True   
        )
        return grad_grad_tuple

    def memory_dataloader(self, batch_size):
        while True:
            MMREDataset.update(self.n_task)
            length = len(self.train_dataset)
            idx_sets = torch.randperm(length)
            cnt = 0
            while cnt < length:
                MMREDataset.update(self.n_task)
                data_store = []
                num_element = 0     
                for i in range(cnt, cnt+batch_size):
                    if i < length:  
                        if num_element==0:  
                            num_element = len(self.train_dataset[idx_sets[i]]) 
                            for j in range(num_element):
                                data_store.append([])
                        j = 0
                        for element in self.train_dataset[idx_sets[i]]:
                            data_store[j].append(element.unsqueeze(0))
                            j += 1
                    else:          
                        break
                for i in range(num_element):
                    data_store[i] = torch.cat(data_store[i], dim=0)
                yield data_store
                cnt += batch_size

            # ------------------------------------------------------------------------------------------------------------------

    def train(self):
        # Author ------------------------------------------------
        GradKeyAndBias.start_layer = START_SHARE_LAYER
        GradKeyAndBias.end_layer = 11  
        GradKeyAndBias.init()
        BertSelfAttention.MODIFY = self.args.do_text_modify     
        # -----------------------------------------------------
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        prgbar = tqdm(total=self._cal_prgbar_times("train"), postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=0)
        self.pbar = prgbar
        lbound = 0
        ubound = self.n_task
        if self.args.do_task_finetune:
            lbound = self.args.task_id
            ubound = self.args.task_id + 1
            print(f"***** Task {lbound+1} Finetune *****")
        for task_id in range(lbound, ubound):
            self.logger.info(f"***** Training at task {task_id+1} *****")
            MMREDataset.update(task_id)  
            for epoch in range(1, self.args.num_epochs+1):
                prgbar.set_description_str(desc="Task {}/{},Epoch {}/{}".format(task_id+1, self.n_task, epoch, self.args.num_epochs))
                avg_loss = 0
                # memory_dl = None
                # if task_id!=0:      # memory_dataloader
                #     memory_dl = self.memory_dataloader(batch_size=1)
                for batch in self.train_data:
                    # if memory_dl != None:
                        # buffer_batch = next(memory_dl)
                        # MMREDataset.update(task_id)     
                        # num_element = len(buffer_batch)
                        # for i in range(num_element):
                        #     batch[i] = torch.cat((batch[i], buffer_batch[i]), dim=0)
                        # print("check!")
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    if task_id == 0:
                        (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None)      
                    elif task_id!=0:
                        if self.args.do_ewc:       
                            (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None)
                            loss += self.ewc_loss()
                        else:
                            loss, coeff_v, coeff_t = self.attention_loss(batch=batch, task_id=None)
                    self.balance_recorder["current"][task_id].append(UnimoREModel.RATIO_VISION)
                    avg_loss += loss.detach().cpu().item()
                    loss.backward()
                    # Author ---------------------------------------------------------------------------------------------
                    if self.args.do_balance:
                        self.balance(coeff_t=coeff_t, coeff_v=coeff_v)
                    # --------------------------------------------------------------------------------------------------
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                    # if epoch==self.args.num_epochs:                           
                    #     GradKeyAndBias.store_param()
                    prgbar.update(1)
                    self.clear()                                              
                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        prgbar.update(self.refresh_step)
                        prgbar.set_postfix_str(print_output)
                        if self.writer is not None:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch, task_id)   # generator to dev.
                    self.test(epoch, task_id)
                self.balance_recorder["current"][task_id].append("SEP") 
            if self.args.do_replay:   
                self.store_buffer()

            if task_id!=0 and self.args.do_replay:
                MMREDataset.update(self.n_task)   
                avg_loss = 0
                for epoch in range(1, self.args.num_epochs+1):
                    prgbar.set_description_str(desc="Replay:Task {}/{},Epoch {}/{}".format(task_id+1, self.n_task, epoch, self.args.num_epochs))
                    MMREDataset.update(self.n_task)  
                    for batch in self.train_data:
                        batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                        (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train", task_id=None)  
                        self.balance_recorder["replay"][task_id].append(UnimoREModel.RATIO_VISION)  
                        self.clear()
                        AttentionReg.clear_attention_list()
                        avg_loss += loss.detach().cpu().item()
                        loss.backward()
                        # Author ---------------------------------------------------------------------------------------------
                        if self.args.do_balance:
                            self.balance(coeff_t=coeff_t, coeff_v=coeff_v)
                        # --------------------------------------------------------------------------------------------------
                        self.optimizer.step()
                        # self.scheduler.step()
                        self.optimizer.zero_grad()
                        prgbar.update(1)
                        self.clear()  
                        if self.step % self.refresh_step == 0:
                            avg_loss = float(avg_loss) / self.refresh_step
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                            prgbar.update(self.refresh_step)
                            prgbar.set_postfix_str(print_output)
                            if self.writer is not None:
                                self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                       global_step=self.step)  # tensorbordx
                            avg_loss = 0
                    if epoch >= self.args.eval_begin_epoch:
                        self.evaluate(epoch, task_id)  # generator to dev.
                        self.test(epoch, task_id)
                    self.balance_recorder["replay"][task_id].append("SEP") 
                MMREDataset.update(task_id)        
            if self.args.do_ewc:
                self.ewc_end_task()
            else:
                self.attention_end_task(task_id=None)
            self.optimizer.zero_grad()
            if self.args.do_save:
                self.save(task_id)


        prgbar.close()
        self.pbar = None
        self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
        self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))
        if self.args.do_balance:
            torch.save(self.balance_recorder, "use_balance.pth")
        else:
            torch.save(self.balance_recorder, "no_use_balance.pth")
        '''
        #with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    # Author: (loss, logits), labels = self._step(batch, mode="train")
                    (loss, logits, coeff_v, coeff_t), labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()
                    loss.backward()
                    # Author ---------------------------------------------------------------------------------------------
                    if self.args.do_balance:
                        self.balance(coeff_t=coeff_t, coeff_v=coeff_v)
                    # --------------------------------------------------------------------------------------------------
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer is not None:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
                    self.test(epoch)
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))
        '''

    def evaluate(self, epoch, task_id):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        MMREDataset.update(task_id)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    (loss, logits, _, _), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                # sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values()), target_names=list(self.re_dict.keys()), digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1 # update best metric(f1 score)
                    if self.args.save_path is not None: # save model
                        pass
                        # torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        # self.logger.info("Save best model at {}".format(self.args.save_path))
        

        self.model.train()

    def test(self, epoch, task_id):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        MMREDataset.update(task_id)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    (loss, logits, _, _), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    
                    pbar.update()
                # evaluate done
                pbar.close()
                # sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values()), target_names=list(self.re_dict.keys()), digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                if self.writer is not None:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx
                total_loss = 0
                ############
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}"\
                            .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch, micro_f1, acc))
                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.best_test_metric = micro_f1
                    self.best_test_epoch = epoch
                    
        self.model.train()
        
    def _step(self, batch, mode="train", task_id=None):
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs, task_id=task_id)
        return outputs, labels

    def before_multimodal_train(self):
        if self.args.do_task_finetune:
            print("***** Start Load Model *****")
            self.load_parameters(self.args.task_id - 1)
        optimizer_grouped_parameters = []
        params = {'lr':self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if self.args.do_froze:                     
                current_name = name.split('.')
                if len(current_name)>=4:
                    _ = self._judge_number(current_name[3])
                    if _ != -1 and _ >=0 and _<self.args.froze_layer:
                        param.requires_grad = False
                        continue                     
            if 'model' in name:
                params['params'].append(param)
                # print(type(param))
            # Author:
            else:
                params['params'].append(param)
            print(name)

        params_share_bias = {'lr':self.args.bias_lr, 'weight_decay':1e-2}
        params_share_key = {'lr':self.args.key_lr, 'weight_decay':1e-2}
        params_share_key['params'] = []
        params_share_bias['params'] = []

        print("Author:", f"bias:{params_share_bias['lr']}, key:{params_share_key['lr']}")
        # params_share = {'lr':1e-6, 'weight_decay':1e-2}     
        for t in ['key','vision_bias', 'text_bias']:
            for i in range(ShareKey.layer):
                # name = f'model.encoder.share.{i}.{t}'
                # print(name)
                if t=='key':
                    params_share_key['params'].append(eval(f"ShareKey.{t}")[i])
                elif 'bias' in t:
                    params_share_bias['params'].append(eval(f"ShareKey.{t}")[i])
                else:
                    assert False
                # params_share['params'].append(eval(f"ShareKey.{t}")[i])
                # print(type(eval(f"ShareKey.{t}")[i]))
                # print(params[name][0][0][0])

        optimizer_grouped_parameters.append(params)
        optimizer_grouped_parameters.append(params_share_key)
        optimizer_grouped_parameters.append(params_share_bias)

        self.optimizer = optim.AdamW(optimizer_grouped_parameters) # ,lr=self.args.lr)
        # print(self.optimizer.param_groups)
        # self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
        #                                                     num_warmup_steps=self.args.warmup_ratio*self.train_num_steps,
        #                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
        # for name, par in self.model.named_parameters():
        #     print(name, par.requires_grad)