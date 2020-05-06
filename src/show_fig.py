import keras
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pathlib


class show(keras.callbacks.Callback):
    def __init__(self, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.tests = [np.load('../patch_data/town/full_h.npy')[np.newaxis],
                        np.load('../patch_data/town/full_v.npy')[np.newaxis]]
        self.output_dir = output_dir
        pathlib.Path(f'{self.output_dir}/fig').mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs={}):
        decoded_imgs = self.model.predict(self.tests, verbose=2)
        plt.grid(False)
        plt.imshow(decoded_imgs[0])
        plt.savefig(f'{self.output_dir}/fig/figure_{epoch:06}.png')
    
    # def on_batch_end(self, batch, logs={}):
    #     decoded_imgs = self.model.predict(self.tests, verbose=2)
    #     print(decoded_imgs[0,50:80,50:80])