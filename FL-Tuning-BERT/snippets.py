#! -*- coding: utf-8 -*-
# CLUE评测
# 模型配置文件

import os
from bert4keras.backend import K
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_gradient_accumulation, extend_with_weight_decay, extend_with_piecewise_linear_lr
from model import PtuningV2AddFNNPtuning

# 通用参数
data_path = './clue_dataset/'

# 权重目录
if not os.path.exists('weights'):
    os.mkdir('weights')

# 输出目录
if not os.path.exists('results'):
    os.mkdir('results')

# 模型路径
config_path = '/data/PretrainedModels/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/PretrainedModels/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/PretrainedModels/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + ["[unused%d]" % i for i in range(1, 100)],
)

# 建立分词器
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# 预训练模型
def load_model():
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

    # 关闭大部分参数
    for layer in m.model.layers:
        if "AddFFN" not in layer.name and "PromptEmbedding" not in layer.name:
            layer.trainable = False
        else:
            pass
        
    return m

base = load_model()


# 模型参数
last_layer = 'Transformer-%s-FeedForward-Norm' % (base.num_hidden_layers - 1)

# 优化器
def get_optimizer(
    learning_rate,
    num_warmup_steps,
    num_train_steps,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=1,
):

    optimizer = extend_with_weight_decay(Adam)
    optimizer = extend_with_piecewise_linear_lr(optimizer)

    optimizer_params = {
        'learning_rate': learning_rate,
        'lr_schedule': {
            num_warmup_steps * grad_accum_steps: 1.0,
            num_train_steps * grad_accum_steps: 0.0,
        },
        'weight_decay_rate': weight_decay_rate,
        'exclude_from_weight_decay': exclude_from_weight_decay,
    }

    if grad_accum_steps > 1:
        optimizer = extend_with_gradient_accumulation(optimizer, name='AdamWG')
        optimizer_params['grad_accum_steps'] = grad_accum_steps

    return optimizer(**optimizer_params)
