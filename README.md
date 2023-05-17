# LMC
Code and dataset for paper [Continual Multimodal Knowledge Graph Construction](https://arxiv.org/pdf/2305.08698v1.pdf).

## 1. Overview

### 1.1 File Tree

Please note "we provide". This is the data set of lifelong benchmark provided by us.

```python
MKGFormer
 |-- MNER	# Multimodal Named Entity Recognition
 |    |-- data          # task data
 |    |    |-- twitter2017
 |    |    |    |-- twitter17_detect            # rcnn detected objects
 |    |    |    |-- twitter2017_aux_images      # visual grounding objects
 |    |    |    |-- twitter2017_images          # raw images
 |    |    |    |-- train.txt                   # text data
 |    |    |    |-- ...
 |    |    |    |-- twitter2017_train_dict.pth  # {imgname: [object-image]}
 |    |    |    |-- sep-PLOM					   # we provide, PER->LOC->ORG->MISC
 |    |    |    |-- sep-POLM-NEW				# we provide, PER->ORG->LOC->MISC
 |    |-- models        # mner model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- utils
 |    |-- run.py
 |-- MRE    # Multimodal Relation Extraction
 |    |-- data          # task data
 |    |    |-- img_detect   # rcnn detected objects
 |    |    |-- img_org      # raw images
 |    |    |-- img_vg       # visual grounding objects
 |    |    |-- txt          # text data
 |    |    |    |-- ours_train.txt
 |    |    |    |-- ours_val.txt
 |    |    |    |-- ours_test.txt
 |    |    |    |-- mre_train_dict.pth  # {imgid: [object-image]}
 |    |    |    |-- sep5				 # we provide, 5 tasks
 |    |    |    |-- sep7				 # we provide, 7 tasks
 |    |    |    |-- sep10				# we provide, 10 tasks
 |    |    |-- vg_data      # [(id, imgname, noun_phrase)], not useful
 |    |    |-- ours_rel2id.json         # relation data
 |    |-- models        # mre model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- run.py
```

### 1.2 Installation

```shell
pip install -r requirements.txt
```



## 2. Continual MRE
In the continual MRE task, we propose a dataset under the continual learning scenario based on the [MEGA](https://github.com/thecharm/Mega) dataset. In view of the reason that the size of the original dataset is too large, we have established data segmentation under different tasks based on the id of the original data in this reponsitory. The data segmentation file of the lifelong benchmark we provide is in `sep5`(5 tasks in `sep5`, 7 tasks in `sep7`, 10 tasks in `sep10`) under the directory `/MRE/data/txt/`. We show how to download the dataset and how to combine the downloaded dataset with the data segmentation we provide in `2.1`. We show how to run code in continual MRE in `2.2`.

### 2.1 Data Download

The original dataset comes from [MEGA](https://github.com/thecharm/Mega). You can download the MRE dataset with detected visual objects using following command:

```shell
cd MRE
wget 120.27.214.45/Data/re/multimodal/data.tar.gz
tar -xzvf data.tar.gz
```

You will get a folder named `data`. Please merge the obtained folder `data` with the folder `MRE/data` provided by us. The merged file tree can refer to section 1.1.



### 2.2 How to Run

You can use following command:

```shell
python -u run.py  --model_name='bert'  --vit_name='openai/clip-vit-base-patch32' --dataset_name='MRE'  --bert_name='bert-base-uncased' --num_epochs=10 --batch_size=2 --lr=0.00001  --eval_begin_epoch=1  --seed=1234 --do_train  --max_seq=80  --prompt_len=4  --aux_size=128  --rcnn_size=64 --gamma=1 --do_balance --alpha=0.1  --save_path='ckpt' --type_text='width' --type_vision='width' --key_lr=0.000001 --bias_lr=0.00001 --task_number=10 --notes='sep10'  --do_replay --do_random	
```

If you want to run in different task, you can set the `--task_number` to 5/7/10.



## 3. Continual MNER
In the continual MNER task, we propose a dataset under the continual learning scenario based on the Twitter2017 datasets. In view of the reason that the size of the original dataset is too large, we have established data segmentation under different tasks based on the id of the original data in this reponsitory. The data segmentation file of the lifelong benchmark we provide is in `sep-PLOM`(`sep-PLOM` is in the order of `PER->LOC->ORG->MISC`, `sep-PLOM-NEW` is in the order of `PER->ORG->LOC->MISC`) under the directory `/MNER/data/twitter2017/`. We show how to download the dataset and how to combine the downloaded dataset with the data segmentation we provide in `3.1`. We show how to run code in continual MRE in `3.2`.

### 3.1 Data Download

You can download the twitter2017 dataset via this [link](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing). Please merge the folder `MNER/data/twitter2017` we provided with the extracted folder. The merged file tree can refer to section 1.1.



### 3.2 How to Run

You can use following command:

```shell
python -u run.py --model_name='bert' --dataset_name='twitter17' --bert_name='bert-base-uncased' --vit_name='openai/clip-vit-base-patch32' --num_epochs=10 --batch_size=2 --lr=0.00003 --warmup_ratio=0 --eval_begin_epoch=1 --seed=1234 --do_train --ignore_idx=0 --max_seq=128 --aux_size=128 --rcnn_size=64 --do_balance --alpha=0.3 --gamma=0.1 --key_lr=0.000003 --bias_lr=0.00003 --crf_lr=0.05 --task_number=4 --notes='sep5' --type_text='width' --type_vision='width' --do_replay --do_random
```


## Citation
If you use the code, please cite the following paper:
```bibtex
@misc{chen2023continual,
      title={Continual Multimodal Knowledge Graph Construction}, 
      author={Xiang Chen and Jintian Zhang and Xiaohan Wang and Tongtong Wu and Shumin Deng and Yongheng Wang and Luo Si and Huajun Chen and Ningyu Zhang},
      year={2023},
      eprint={2305.08698},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```






