from keras.layers import Input, Conv2D, Conv3D, ZeroPadding3D, Lambda, Concatenate
from keras.models import Model

# GPU memory settings----------------------
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)
# -----------------------------------------

def conv3D_branch(x):
    x = Lambda(lambda x: x - K.mean(x, axis=(0,1,2,3)))(x)
    x = ZeroPadding3D(padding=(0, 4, 4))(x)
    for n_filters in [32, 64, 64, 64]:
        x = Conv3D(n_filters, kernel_size=(3, 3, 3), padding='valid', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x

def build_model():
    # 3Dconv branch1
    inputs1 = Input(shape=((9, None, None, 3)), name='inputs1')
    processed1 = conv3D_branch(inputs1)
    # 3Dconv branch2
    inputs2 = Input(shape=((9, None, None, 3)), name='inputs2')
    processed2 = conv3D_branch(inputs2)

    # concat -> 2D conv
    x = Concatenate()([processed1, processed2])
    for n_filters in [64, 32, 32, 16]:
        x = Conv2D(n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Conv2D(1, kernel_size=3, padding='same', kernel_initializer='glorot_uniform')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=3))(x)

    return Model(inputs=[inputs1, inputs2], outputs=x)