import tensorflow as tf
from git.RecognitionofSignLanguage.utils.Transformer_utils import Transformer_Utils as tu


class Transformer(tf.keras.Model):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks

    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(tu.UNITS, 8))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(tu.UNITS * tu.MLP_RATIO, activation=tu.GELU, kernel_initializer=tu.INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(tu.MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(tu.UNITS, kernel_initializer=tu.INIT_HE_UNIFORM),
            ]))

    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)

        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(self.scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

    def scaled_dot_product(self, q, k, v, softmax, attention_mask):
        # calculates Q . K(transpose)
        qkt = tf.matmul(q, k, transpose_b=True)
        # caculates scaling factor
        dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
        scaled_qkt = qkt / dk
        softmax = softmax(scaled_qkt, mask=attention_mask)

        z = tf.matmul(softmax, v)
        # shape: (m,Tx,depth), same shape as q,k,v
        return z