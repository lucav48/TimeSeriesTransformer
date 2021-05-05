from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from transformer.component.MultiAttention import MultiAttention
from transformer.component.SingleAttention import SingleAttention
from transformer.component.Time2Vector import Time2Vector
from transformer.component.TransformerEncoder import TransformerEncoder


def create_model(seq_len=128, seq_dim=5, n_heads=12, d_k=32, d_v=32, ff_dim=1024,
                 n_attention=3):
    '''Initialize time and transformer layers'''
    time_embedding = Time2Vector(seq_len)
    # attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    # attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    # attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, seq_dim))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    for i in range(n_attention):
        x = TransformerEncoder(d_k, d_v, n_heads, ff_dim)((x, x, x))
    # x = attn_layer1((x, x, x))
    # x = attn_layer2((x, x, x))
    # x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(seq_len, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=in_seq, outputs=out)
    return model

def load_custom_model(path):
    return load_model(path,
                      custom_objects={'Time2Vector': Time2Vector,
                                      'SingleAttention': SingleAttention,
                                      'MultiAttention': MultiAttention,
                                      'TransformerEncoder': TransformerEncoder})
