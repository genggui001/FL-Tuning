#! -*- coding:utf-8 -*-
# CLUE评测
# tnews文本分类
# 思路：取[CLS]然后接Dense+Softmax分类

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["TF_KERAS"] = '1'

import json
import numpy as np
from snippets import *
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm

# 基本参数
labels = [
    '100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112',
    '113', '114', '115', '116'
]
num_classes = len(labels)
maxlen = 128
batch_size = 32
epochs = 30
learning_rate = 5e-5


def load_data(filename):
    """加载数据
    格式：[(文本, 标签id)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l.get('label', '100')
            D.append((text, labels.index(label)))
    return D


# 加载数据集
train_data = load_data(data_path + 'tnews/train.json')
valid_data = load_data(data_path + 'tnews/dev.json')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 构建模型
output = base.model.get_layer(last_layer).output
output = keras.layers.Lambda(lambda x: x[:, 0])(output)
output = keras.layers.Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=base.initializer
)(output)

model = keras.models.Model(base.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=get_optimizer(
        learning_rate=learning_rate,
        num_warmup_steps=0,
        num_train_steps=len(train_generator)*epochs,
    ),
    metrics=['accuracy']
)


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('weights/tnews.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)

    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            l = json.dumps({'id': str(l['id']), 'label': labels[r]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    # Train
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    # Test
    model.load_weights('weights/tnews.weights')
    test_predict(
        in_file=data_path + 'tnews/test1.0.json',
        out_file='results/tnews10_predict.json'
    )
    test_predict(
        in_file=data_path + 'tnews/test.json',
        out_file='results/tnews11_predict.json'
    )

else:

    model.load_weights('weights/tnews.weights')
