import torch
from torch import optim
from torch.optim.sgd import SGD
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from utils.ner_evaluate import evaluate, evaluate_each_class
from seqeval.metrics import classification_report
from models.modeling_unimo_mike import ShareKey, START_SHARE_LAYER, AttentionReg, BertSelfAttention
from processor.datasets import MMPNERBertDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F

sharekey_dict = [
    "config", "v_max_l", "t_max_l", "num_heads", "head_dim", "key", "vision_bias", "text_bias",
    "text_bn", "vision_bn", "layer", "device"
]
attentionreg_dict = [
    "old_model", "old_key", "old_text_bias", "old_vision_bias",
    "attention_text_list", "attention_vision_list", "zero"
]

class BertTrainer(object):
    def __init__(self, train_data=None, train_dataset=None, dev_data=None, test_data=None, model=None, process=None, label_map=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data        # dataloader
        self.train_dataset = train_dataset  # dataset
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.process = process
        self.logger = logger
        self.label_map = label_map
        self.label_map_inverse = {self.label_map[key]:key for key in self.label_map}    # Author Add
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.multiModal_before_train()

        self.n_task = self.args.task_number
        self.data_length = []   # Count the number of samples for each task
        self._statistic_data_length()

    # Author----------------------------------------------------------------------------------------------------

    # tools
    def _statistic_data_length(self):
        self.train_num_steps = 0
        class_per_task = self.model.num_labels // self.n_task + min(1, self.model.num_labels % self.n_task)
        cur_buffer_num = 0
        for i in range(self.n_task):
            MMPNERBertDataset.update(i)
            self.data_length.append(len(self.train_dataset))
            self.train_num_steps += (self.data_length[i] // self.args.batch_size + min(1, self.data_length[i] % self.args.batch_size))
            if self.args.do_replay and i!=0:
                cur_buffer_num += class_per_task * self.args.buffer_size
                self.train_num_steps += cur_buffer_num // self.args.batch_size
        self.train_num_steps *= self.args.num_epochs
        print(f"Task Number:{self.n_task}, length:{self.data_length}")

    def _judge_number(self, number):
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        for n in numbers:
            if number == n:
                return int(n)
        return -1

    # balance
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
            # layer = str(name).split('.')[1].lower() #
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

    # attention loss
    def clear(self):
        AttentionReg.attention_text_list.clear()
        AttentionReg.attention_vision_list.clear()

    def end_task(self):
        # save the current model
        AttentionReg.update_old_model(self.model)
        AttentionReg.update_old_key_and_bias()

    def attention_loss(self, batch, return_logits=False, return_labels=False):
        # 1. get old attention map from old model
        AttentionReg.clear_attention_list()
        with torch.no_grad():
            AttentionReg.change_ShareKey() 
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs = batch
            AttentionReg.old_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels,
                images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs
            )
            old_vision_attention = AttentionReg.attention_vision_list
            old_text_attention = AttentionReg.attention_text_list
        # 2.get current attention map now
        AttentionReg.change_ShareKey()  
        AttentionReg.clear_attention_list()
        loss, logits, coeff_v, coeff_t = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
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
            return attention_mask, loss, logits, coeff_v, coeff_t, labels
        elif return_labels==False and return_logits==True:
            assert False

    # buffer
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
                "10.attention.self.key." in name or \
                "11.attention.self.key." in name or \
                "text_pooler" in name or \
                "text_classifier.dense" in name or \
                "vision_classifier.dense" in name:
            return False
        return True

    def store_buffer(self, use_attention_loss=False):
        new_data = []
        buffer_size = self.args.buffer_size
        # 1. get scores {labels:[{}, {}, {}], labels:[]}
        candidates = self.compute_train_score(
            use_attention_loss=use_attention_loss,
            random_sample=self.args.do_random
        )
        # 2. sort 
        print("replay labels:", candidates.keys())
        for label in candidates.keys():
            lists = candidates[label]
            lists.sort(key=lambda x: x[list(x.keys())[1]], reverse=True)  
            cnt = 0
            for item in lists:
                cnt += 1
                new_data.append(self.train_dataset.get_data_for_adding(task_id=MMPNERBertDataset.ID, idx=item["index"]))
                if cnt >= buffer_size:  # full
                    break
        # 3. add to buffer
        self.train_dataset.adding_to_buffer(new_data=new_data)

    def parse_label(self, labels):
        # labels: [n]
        # return label
        for idx in range(labels.shape[0]):
            label_id = labels[idx].item()
            if "-" in self.label_map_inverse[label_id]:
                return self.label_map_inverse[label_id].split("-")[1]
        return "none"

    def compute_train_score(self, use_attention_loss=False, random_sample=True, task_id=None):
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
        self.model.eval()
        output_collections = {}  # label: [{}, {}, {}]
        pbar = tqdm(total=self.data_length[MMPNERBertDataset.ID])
        for idx, inputs in enumerate(train_dataloader):
            pbar.update(1)
            # print(inputs[0][0], self.train_dataset[idx][0])
            if idx >= self.data_length[MMPNERBertDataset.ID]:  
                break
            inputs_ = inputs.copy()
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs)
            if use_attention_loss == False:
                attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="train") 
            elif use_attention_loss == True:
                attention_mask, loss, logits, coeff_v, coeff_t, labels = self.attention_loss(batch=batch,
                                                                             return_logits=True, return_labels=True)
            if random_sample:  
                label = self.parse_label(labels[0].view(-1))
                outputs = {"index": idx, "influence": torch.rand([1]).item(), "label": label}
                _label = label
                if _label not in output_collections.keys():
                    output_collections[_label] = []
                output_collections[_label].append(outputs)
                self.clear()
                AttentionReg.clear_attention_list()
                torch.cuda.empty_cache()
                continue
            prob = F.softmax(logits, dim=-1)  # [bs, len, n_label]
            # labels: [bs, len]
            prob_gt = torch.gather(prob, 2, labels.unsqueeze(2))

            self.model.zero_grad()
            self.optimizer.zero_grad()

            model_parameters = [param for name, param in self.model.named_parameters() if
                                param.requires_grad and self._judge_use_param(name)]
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
            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in inputs_)
            if use_attention_loss == False:
                attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="train")  
            elif use_attention_loss == True:
                attention_mask, loss, logits, coeff_v, coeff_t, labels = self.attention_loss(batch=batch, task_id=None,
                                                                             return_logits=True, return_labels=True)
            self.model.zero_grad()
            self.optimizer.zero_grad()

            model_parameters_ = [param for name, param in self.model.named_parameters() if
                                 param.requires_grad and self._judge_use_param(name)]
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
            _label = self.parse_label(labels[0].view(-1))
            outputs = {"index": idx, "influence": influence, "label": _label}
            if _label not in output_collections.keys():
                output_collections[_label] = []
            output_collections[_label].append(outputs)
            del v, s, grad_tuple_, inputs_, train_dataloader_, influence, model_parameters_, model_parameters, prob, prob_gt, batch
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
                if n > num_samples:
                    break
        inverse_hvp = [X / scale for X in last_estimate]
        del this_estimate, inputs_, last_estimate, batch, batch_  # Author add
        torch.cuda.empty_cache()
        return inverse_hvp

    def compute_hessian_vector_products(self, batch, vectors, use_attention_loss=False):
        if use_attention_loss == False:
            attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="train")  
        elif use_attention_loss == True:
            loss, coeff_v, coeff_t = self.attention_loss(batch=batch)

        self.model.zero_grad()
        self.optimizer.zero_grad() 

        model_parameters = [param for name, param in self.model.named_parameters() if
                            param.requires_grad and self._judge_use_param(name)]
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

        model_parameters_ = [param for name, param in self.model.named_parameters() if
                             param.requires_grad and self._judge_use_param(name)]
        for i in range(START_SHARE_LAYER, ShareKey.layer):
            model_parameters_.append(ShareKey.key[i])
            model_parameters_.append(ShareKey.text_bias[i])
            model_parameters_.append(ShareKey.vision_bias[i])

        grad_grad_tuple = torch.autograd.grad(
            outputs=grad_tuple,
            inputs=model_parameters_,
            grad_outputs=vectors,
            only_inputs=True,
            allow_unused=True 
        )
        return grad_grad_tuple

    # save and load

    def load_parameters(self, task_id):
        self.model.load_state_dict(torch.load(self.args.save_path + f"/model_{task_id}.pth"))
        share_parameters = torch.load(self.args.save_path + f"/share_{task_id}.pth")
        print("***** Start Load *****")
        print(f"***** Load from {self.args.save_path}/share_{task_id}.pth, model_{task_id}.pth *****")
        for name in sharekey_dict:
            exec(f"ShareKey.{name} = share_parameters['ShareKey'][name]")
        for name in attentionreg_dict:
            exec(f"AttentionReg.{name}=share_parameters['AttentionReg'][name]")

    def save(self, task_id=-1):
        torch.save(self.model.state_dict(), self.args.save_path + f"/model_{task_id}.pth")
        share_parameters = {"ShareKey": {}, "AttentionReg": {}}
        for name in sharekey_dict:
            share_parameters["ShareKey"][name] = eval(f"ShareKey.{name}")
        for name in attentionreg_dict:
            share_parameters["AttentionReg"][name] = eval(f"AttentionReg.{name}")
        torch.save(share_parameters, self.args.save_path + f"/share_{task_id}.pth")
        self.logger.info("Save best model at {}".format(self.args.save_path))

    # ----------------------------------------------------------------------------------------------------

    def train(self, clip_model_dict=None, bert_model_dict=None):
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

        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        cnt = 0
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
                    cnt += 1
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
                    cnt += 1
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        # Author ----------------------------------------------------------------------------------------------
        BertSelfAttention.MODIFY = self.args.do_text_modify    
        lbound = 0
        ubound = self.n_task
        if self.args.do_task_finetune:
            lbound = self.args.task_id
            ubound = self.args.task_id + 1
            print(f"***** Task {lbound+1} Finetune *****")
        # ---------------------------------------------------------------------------------------------------
            
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            for task_id in range(lbound, ubound):
                self.logger.info(f"***** Training at task {task_id + 1} *****")
                MMPNERBertDataset.update(task_id)
                for epoch in range(1, self.args.num_epochs + 1):
                    self.pbar.set_description_str(desc="Task {}/{},Epoch {}/{},Not Replay".format(task_id+1, self.n_task, epoch, self.args.num_epochs))
                    avg_loss = 0
                    y_true, y_pred = [], []
                    y_true_idx, y_pred_idx = [], []
                    for batch in self.train_data:
                        self.step += 1
                        batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                        if task_id == 0:
                            attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="train")
                        elif task_id != 0:
                            attention_mask, loss, logits, coeff_v, coeff_t, labels = self.attention_loss(batch=batch, return_logits=True, return_labels=True)
                        avg_loss += loss.detach().cpu().item()

                        loss.backward()
                        # Author -------------------------------------
                        if self.args.do_balance:
                            self.balance(coeff_t=coeff_t, coeff_v=coeff_v)
                        # ------------------------------------------

                        self.optimizer.step()
                        # self.scheduler.step()
                        self.optimizer.zero_grad()
                        # self.pbar.update(1)
                        self.clear()

                        if isinstance(logits, torch.Tensor):
                            logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                        label_ids = labels.to('cpu').numpy()
                        input_mask = attention_mask.to('cpu').numpy()
                        label_map = {idx:label for label, idx in self.label_map.items()}
                        for i, mask in enumerate(input_mask):
                            temp_1 = []
                            temp_2 = []
                            temp_1_idx, temp_2_idx = [], []
                            for j, m in enumerate(mask):
                                if j == 0:
                                    continue
                                if m:
                                    if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                        temp_1.append(label_map[label_ids[i][j]])
                                        temp_2.append(label_map[logits[i][j]])
                                        temp_1_idx.append(label_ids[i][j])
                                        temp_2_idx.append(logits[i][j])
                                else:
                                    break
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            y_true_idx.append(temp_1_idx)
                            y_pred_idx.append(temp_2_idx)

                        if self.step % self.refresh_step == 0:
                            avg_loss = float(avg_loss) / self.refresh_step
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                            pbar.update(self.refresh_step)
                            pbar.set_postfix_str(print_output)
                            if self.writer:
                                self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                            avg_loss = 0
                    results = classification_report(y_true, y_pred, digits=4)
                    self.logger.info("***** Train Eval results *****")
                    self.logger.info("\n%s", results)
                    f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                    if self.writer:
                        self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                    self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}."\
                                .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch, f1_score))
                    if f1_score > self.best_train_metric:
                        self.best_train_metric = f1_score
                        self.best_train_epoch = epoch
                    if epoch >= self.args.eval_begin_epoch:
                        self.evaluate(epoch, task_id)   # generator to dev.
                        self.test(epoch, task_id)
                torch.cuda.empty_cache()

                if self.args.do_replay:
                    self.store_buffer()

                if task_id != 0 and self.args.do_replay:
                    MMPNERBertDataset.update(self.n_task)
                    for epoch in range(1, self.args.num_epochs + 1):
                        avg_loss = 0
                        y_true, y_pred = [], []
                        y_true_idx, y_pred_idx = [], []
                        self.pbar.set_description_str(desc="Task {}/{},Epoch {}/{}, Replaying".format(task_id + 1, self.n_task, epoch, self.args.num_epochs))
                        MMPNERBertDataset.update(self.n_task)

                        for batch in self.train_data:
                            self.step += 1
                            batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                            attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="train")
                            avg_loss += loss.detach().cpu().item()
                            self.clear()
                            AttentionReg.clear_attention_list()
                            loss.backward()
                            # Author -------------------------------------
                            if self.args.do_balance:
                                self.balance(coeff_t=coeff_t, coeff_v=coeff_v)
                            # ------------------------------------------
                            self.optimizer.step()
                            # self.scheduler.step()
                            self.optimizer.zero_grad()
                            # self.pbar.update(1)

                            if isinstance(logits, torch.Tensor):
                                logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                            label_ids = labels.to('cpu').numpy()
                            input_mask = attention_mask.to('cpu').numpy()
                            label_map = {idx: label for label, idx in self.label_map.items()}
                            for i, mask in enumerate(input_mask):
                                temp_1 = []
                                temp_2 = []
                                temp_1_idx, temp_2_idx = [], []
                                for j, m in enumerate(mask):
                                    if j == 0:
                                        continue
                                    if m:
                                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                            temp_1.append(label_map[label_ids[i][j]])
                                            temp_2.append(label_map[logits[i][j]])
                                            temp_1_idx.append(label_ids[i][j])
                                            temp_2_idx.append(logits[i][j])
                                    else:
                                        break
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                y_true_idx.append(temp_1_idx)
                                y_pred_idx.append(temp_2_idx)

                            if self.step % self.refresh_step == 0:
                                avg_loss = float(avg_loss) / self.refresh_step
                                print_output = "loss:{:<6.5f}".format(avg_loss)
                                pbar.update(self.refresh_step)
                                pbar.set_postfix_str(print_output)
                                if self.writer:
                                    self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                           global_step=self.step)  # tensorbordx
                                avg_loss = 0
                        results = classification_report(y_true, y_pred, digits=4)
                        self.logger.info("***** Train Eval results *****")
                        self.logger.info("\n%s", results)
                        f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                        if self.writer:
                            self.writer.add_scalar(tag='train_f1', scalar_value=f1_score,
                                                   global_step=epoch)  # tensorbordx
                        self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}." \
                                         .format(epoch, self.args.num_epochs, self.best_train_metric,
                                                 self.best_train_epoch, f1_score))
                        if f1_score > self.best_train_metric:
                            self.best_train_metric = f1_score
                            self.best_train_epoch = epoch
                        if epoch >= self.args.eval_begin_epoch:
                            self.evaluate(epoch, task_id)  # generator to dev.
                            self.test(epoch, task_id)
                    MMPNERBertDataset.update(task_id)
                self.end_task() 
                self.optimizer.zero_grad()
                if self.args.do_save:
                    self.save(task_id)
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch, task_id):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        MMPNERBertDataset.update(task_id)
        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    # attention_mask, labels, logits, loss = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):    
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for i, mask in enumerate(input_mask):
                        temp_1 = []
                        temp_2 = []
                        temp_1_idx, temp_2_idx = [], []
                        for j, m in enumerate(mask):
                            if j == 0:
                                continue
                            if m:
                                if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                    temp_1.append(label_map[label_ids[i][j]])
                                    temp_2.append(label_map[logits[i][j]])
                                    temp_1_idx.append(label_ids[i][j])
                                    temp_2_idx.append(logits[i][j])
                            else:
                                break
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        y_true_idx.append(temp_1_idx)
                        y_pred_idx.append(temp_2_idx)
                    pbar.update()
                # evaluate done
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4)  
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
                if self.writer: 
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/step, global_step=epoch)    # tensorbordx
               
                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, f1_score))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score # update best metric(f1 score)
                    if self.args.save_path is not None: # save model
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
               

        self.model.train()

    def test(self, epoch, task_id):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        MMPNERBertDataset.update(task_id)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    # attention_mask, labels, logits, loss = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    attention_mask, labels, logits, loss, coeff_v, coeff_t = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().tolist()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for i, mask in enumerate(input_mask):
                        temp_1 = []
                        temp_2 = []
                        temp_1_idx, temp_2_idx = [], []
                        for j, m in enumerate(mask):
                            if j == 0:
                                continue
                            if m:
                                if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                    temp_1.append(label_map[label_ids[i][j]])
                                    temp_2.append(label_map[logits[i][j]])
                                    temp_1_idx.append(label_ids[i][j])
                                    temp_2_idx.append(logits[i][j])
                            else:
                                break
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        y_true_idx.append(temp_1_idx)
                        y_pred_idx.append(temp_2_idx)
                    pbar.update()
                # evaluate done
                pbar.close()

                results = classification_report(y_true, y_pred, digits=4) 
                
                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx
                total_loss = 0
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch, f1_score))
                if f1_score >= self.best_test_metric:  # this epoch get best performance
                    self.best_test_metric = f1_score
                    self.best_test_epoch = epoch
                   
        self.model.train()

    def _step(self, batch, mode="train"):
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs = batch
        loss, logits, coeff_v, coeff_t = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
        return attention_mask, labels, logits, loss, coeff_v, coeff_t

    def multiModal_before_train(self):
        if self.args.do_task_finetune:
            print("***** Start Load Model *****")
            self.load_parameters(self.args.task_id - 1)
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        params_crf = {'lr':self.args.crf_lr, 'weight_decay': 1e-2}
        params_crf['params'] = []
        for name, param in self.model.named_parameters():
            if self.args.do_froze:  
                current_name = name.split('.')
                if len(current_name) >= 4:
                    _ = self._judge_number(current_name[3])
                    if _ != -1 and _ >= 0 and _ < self.args.froze_layer:
                        param.requires_grad = False
                        continue  
            if 'crf' in name or 'cat_classifier' in name:
                params_crf['params'].append(param)
                print("crf:", name)
            else:
                params['params'].append(param)
                print("model:", name)

        params_share_bias = {'lr': self.args.bias_lr, 'weight_decay': 1e-2}
        params_share_key = {'lr': self.args.key_lr, 'weight_decay': 1e-2}
        params_share_key['params'] = []
        params_share_bias['params'] = []

        print("Author:", f"bias:{params_share_bias['lr']}, key:{params_share_key['lr']}, crf:{params_crf['lr']}")
        for t in ['key', 'vision_bias', 'text_bias']:
            for i in range(ShareKey.layer):
                if t == 'key':
                    params_share_key['params'].append(eval(f"ShareKey.{t}")[i])
                elif 'bias' in t:
                    params_share_bias['params'].append(eval(f"ShareKey.{t}")[i])
                else:
                    assert False

        optimizer_grouped_parameters.append(params)
        optimizer_grouped_parameters.append(params_share_key)
        optimizer_grouped_parameters.append(params_share_bias)
        optimizer_grouped_parameters.append(params_crf)

        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.model.to(self.args.device)
