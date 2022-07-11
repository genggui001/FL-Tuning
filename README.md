# FL-Tuning
The source code of FL-Tuning, including the model source code and some downstream tasks based on this model.


<img width="750" alt="" src="./model_ architecture.png">

The model details of FL-Tuning are described in the paper:

[FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer](https://arxiv.org/abs/2206.15312)

#  Reproduct


## Directory Structure


## Requirements
### GPU Requirements
* We have successfully run this code on 3090, V100 and A100 respectively.

### Python Requirements
```
pip install nvidia-pyindex==1.0.9
conda env create -f tf1.15.yml
```

## Data
We use the [CLUE benchmark](https://github.com/CLUEbenchmark/CLUE) as our downstream tasks.

The datasets can be downloaded from the [CLUE benchmark](https://github.com/CLUEbenchmark/CLUE) repository. The datasets should be the following format.
```
├── clue_dataset
│   ├── afqmc
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   ├── c3
│   │   ├── d-dev.json
│   │   ├── d-train.json
│   │   ├── m-dev.json
│   │   ├── m-train.json
│   │   ├── README.md
│   │   ├── test1.0.json
│   │   └── test1.1.json
│   ├── chid
│   │   ├── dev_answer.json
│   │   ├── ......
│   ├── ......
│
```

## Pretrained Language Model
We conducted experiments on three language models RoBERTa, NEZHA, and RoFormer respectively. You can download the pretrained language models from the links below.
* [chinese_roberta_wwm_ext_L-12_H-768_A-12](https://github.com/brightmart/roberta_zh)
* [NEZHA-Base-WWM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)
* [chinese_roformer-char_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/roformer)

The directory of the model in **snippets.py** should be changed, for example:
```Python
config_path = '/data/PretrainedModels/chinese_roformer-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/PretrainedModels/chinese_roformer-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/PretrainedModels/chinese_roformer-char_L-12_H-768_A-12/vocab.txt'
```
## Prompt Size
The prompt size is in the **snippets.py** as below.
```Python
m = build_transformer_model(
    config_path, 
    checkpoint_path, 
    model=PtuningV2AddFNNPtuning,
    application='unilm', 
    prompt_size=0,
    ffn_prompt_size=160,
    keep_tokens=keep_tokens,
    return_keras_model=False,
)
```
The hyperparameter `ffn_prompt_size` represents the prompt size of **FL-Tuning**, the FL-Tuning will be disabled when the `ffn_prompt_size` is `0`.

The hyperparameter `prompt_size` represents the prompt size of **P-tuning V2**, the P-tuning V2 will be disabled when the `prompt_size` is `0`.

For the convenience of comparison experiments, we also implemented P-Tuning V2 in this code, you can use these two tuning methods by changing the hyperparameter `ffn_prompt_size` and `prompt_size`. In most of our experiments, the size is `160` (One of the tuning methods is enabled) or `0` (The other tuning method is disabled).

## Run
You can train and test the downstream tasks directly by running the downstream task python files, for example:
```Shell
python afqmc.py
python c3.py
...
```

# Acknowledgement

The code is implemented based on [bert4keras](https://github.com/bojone/bert4keras) and [CLUE-bert4keras](https://github.com/bojone/CLUE-bert4keras). Sincere thanks to the author for his selfless dedication.

# Citation

```
@misc{https://doi.org/10.48550/arxiv.2206.15312,
  doi = {10.48550/ARXIV.2206.15312},
  
  url = {https://arxiv.org/abs/2206.15312},
  
  author = {Liu, Jingping and Song, Yuqiu and Xue, Kui and Sun, Hongli and Wang, Chao and Chen, Lihan and Jiang, Haiyun and Liang, Jiaqing and Ruan, Tong},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {FL-Tuning: Layer Tuning for Feed-Forward Network in Transformer},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
