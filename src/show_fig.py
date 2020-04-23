import keras
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
class show(keras.callbacks.Callback):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.tests = [np.load('../patch_data/town/full_h.npy')[np.newaxis],
                        np.load('../patch_data/town/full_v.npy')[np.newaxis]]

    def on_epoch_end(self, epoch, logs={}):
        decoded_imgs = self.model.predict(self.tests, verbose=2)
        plt.imshow(decoded_imgs[0])
        plt.savefig(f'../output/fig/figure_{epoch:06}.png')