from bert4keras.backend import keras, K
from bert4keras.backend import sequence_masking
from bert4keras.backend import recompute_grad
from bert4keras.layers import Layer, LayerNormalization, integerize_shape
from bert4keras.models import NEZHA
from bert4keras.layers import Embedding, Dropout, Masking, PositionEmbedding, Add, FeedForward
from bert4keras.layers import LayerNormalization, Embedding, BiasAdd
import tensorflow as tf

from keras import initializers, activations
from keras.layers import Input, Dense, Lambda, Reshape, Permute, Activation

infinity = 1e12

class PromptEmbedding(Layer):
    def __init__(
        self, 
        prompt_size,
        heads,
        hidden_size,
        dropout_prob=0.1,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(PromptEmbedding, self).__init__(**kwargs)
        self.prompt_size = prompt_size
        self.heads = heads
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.kernel_initializer = initializers.get(kernel_initializer)
        

    def build(self, input_shape):        
        super(PromptEmbedding, self).build(input_shape)
        
        self.prompt_embedding = self.add_weight(
            name='prompt_embedding',
            shape=(self.prompt_size, self.heads, self.hidden_size),
            initializer=self.kernel_initializer,
        )
        
        self.prompt_mask = K.constant([[True] * self.prompt_size], dtype=bool)
        
        self.dropout = keras.layers.Dropout(
            rate=self.dropout_prob
        )

    def call(self, inputs, mask=None):
        batch_size = K.shape(inputs)[0]
        
        prompt_emb = K.tile(self.prompt_embedding[None,:,:,:],(batch_size, 1, 1, 1))
        prompt_emb = self.dropout(prompt_emb)
        
#         print(prompt_emb)

        return prompt_emb

        
    def compute_output_shape(self, input_shape):
        # h 的形状一样
        return (
            input_shape[0], 
            self.prompt_size, 
            self.heads,
            self.hidden_size
        )
        

    def compute_mask(self, inputs, mask=None):
        batch_size = K.shape(inputs)[0]
        
        prompt_mask = K.tile(self.prompt_mask, (batch_size, 1))
        
#         print(prompt_mask)
        
        return prompt_mask
        

    def get_config(self):
        config = {
            'prompt_size': self.prompt_size,
            'heads': self.heads,
            'hidden_size': self.hidden_size,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(PromptEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        key_size=None,
        use_bias=True,
        attention_scale=True,
        attention_dropout=None,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        assert "input_names" in kwargs, "add input_names in kwargs"
        assert len(kwargs['input_names']) == len(inputs)
        
        inputs = {input_name: inputs_item for input_name, inputs_item in zip(kwargs['input_names'], inputs)}
        input_masks = {input_name + "_mask": mask_item for input_name, mask_item in zip(kwargs['input_names'], mask)}
        
        
        # mask增加
        inputs['q_mask'] = input_masks['q_mask']
        inputs['v_mask'] = input_masks['v_mask']
            
        # 线性变换
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        inputs["qw"] = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        inputs["kw"] = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        inputs["vw"] = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        
        # 修正past_key
        if "past_key" in inputs and "past_value" in inputs:
            past_key = inputs['past_key']
            past_value = inputs['past_value']
            
            past_key_mask = input_masks['past_key_mask']
            past_value_mask = input_masks['past_value_mask']
            
            inputs["kw"] = K.concatenate([past_key, inputs["kw"]], axis=1)
            inputs["vw"] = K.concatenate([past_value, inputs["vw"]], axis=1)
            
            inputs['v_mask'] = K.concatenate([past_value_mask, inputs['v_mask']], axis=1)
            
            if "a_bias" in inputs:
                # [32,1,256,256] to [32,12,256,266] 
                a_bias = inputs["a_bias"]
                
                head_size = K.shape(a_bias)[1]
                s_len = K.shape(a_bias)[2]
                
                a_bias_past_value_mask = -(1 - K.cast(past_value_mask, K.floatx())) * infinity
                a_bias_past_value_mask = a_bias_past_value_mask[:,None,None,:]
                a_bias_past_value_mask = K.tile(a_bias_past_value_mask, (1, head_size, s_len, 1))
                
                a_bias = K.concatenate((a_bias_past_value_mask, a_bias), axis=-1)
                inputs["a_bias"] = a_bias
            
            # 处理位置编码
            if 'typical_relative_p_bias' in inputs:
                position_bias = inputs['typical_relative_p_bias']

                key_len = K.shape(past_key)[1]

                # jkd pad k -> len(past_key) + k
                position_bias_pad = K.zeros_like(position_bias[:,0:1,:])
                position_bias_pad = K.tile(position_bias_pad, (1, key_len, 1))

                position_bias = K.concatenate((position_bias_pad, position_bias), axis=1)
                inputs['typical_relative_p_bias'] = position_bias
            
        # Attention
        o, a = self.pay_attention_to(inputs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的atttention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        qw, kw, vw = inputs["qw"], inputs["kw"], inputs["vw"]
        q_mask, v_mask = inputs["q_mask"], inputs["v_mask"]
        
        a_bias, p_bias = None, None
        
        if "a_bias" in inputs:
            a_bias = inputs["a_bias"]

        if 'rotary_p_bias' in inputs:
            cos_pos = K.repeat_elements(inputs['rotary_p_bias'][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs['rotary_p_bias'][..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        
        # 处理位置编码
        if 'typical_relative_p_bias' in inputs:
            position_bias = inputs['typical_relative_p_bias']
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif 't5_relative_p_bias' in inputs:
            position_bias = K.permute_dimensions(inputs['t5_relative_p_bias'], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)
            
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        
        if 'typical_relative_p_bias' in inputs:
            position_bias = inputs['typical_relative_p_bias']
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PtuningV2AddFNNPtuning(NEZHA):

    def __init__(
        self, 
        prompt_size, ffn_prompt_size, 
        **kwargs
    ):
        super(PtuningV2AddFNNPtuning, self).__init__(**kwargs)
        self.prompt_size = prompt_size
        self.ffn_prompt_size = ffn_prompt_size


    def apply_main_layers(self, inputs, index):
        """NEZHA的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        add_feed_forward_name = 'Transformer-%d-AddFFN' % index

        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi, x, x_names = x, [x, x, x, position_bias], ["q", "k", "v", "typical_relative_p_bias"]

        # PromptEmbedding
        if self.prompt_size > 0:

            p_keys = self.apply(
                inputs=xi,
                layer=PromptEmbedding,
                prompt_size=self.prompt_size,
                heads=self.num_attention_heads,
                hidden_size=self.attention_key_size,
                kernel_initializer=self.initializer,
                name='%s-Key-PromptEmbedding' % attention_name
            )

            p_values = self.apply(
                inputs=xi,
                layer=PromptEmbedding,
                prompt_size=self.prompt_size,
                heads=self.num_attention_heads,
                hidden_size=self.attention_head_size,
                kernel_initializer=self.initializer,
                name='%s-Value-PromptEmbedding' % attention_name
            )
            
            x = x + [p_keys, p_values]
            x_names = x_names + ["past_key", "past_value"]

        if attention_mask is not None:
            x.append(attention_mask)
            x_names.append('a_bias')

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments={
                "input_names": x_names,
            },
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        if self.ffn_prompt_size > 0:
            x_o = self.apply(
                inputs=x,
                layer=FeedForward,
                units=self.intermediate_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=feed_forward_name
            )

            x_a = self.apply(
                inputs=xi,
                layer=FeedForward,
                units=self.ffn_prompt_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=add_feed_forward_name
            )

            x = self.apply(
                inputs=[x_o, x_a],
                layer=Lambda,
                function=lambda x, mask: x[0] + x[1],
                mask=lambda x, mask: mask[0],
                name='%s-Sum' % feed_forward_name
            )
        else:
            x = self.apply(
                inputs=x,
                layer=FeedForward,
                units=self.intermediate_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name=feed_forward_name
            )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

