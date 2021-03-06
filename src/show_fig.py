import keras
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pathlib


class show(keras.callbacks.Callback):
    def __init__(self, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.tests = [np.load('../patch_data/town/full_h.npy')[np.newaxis] / 255.0,
                        np.load('../patch_data/town/full_v.npy')[np.newaxis] / 255.0]
        self.output_dir = output_dir
        pathlib.Path(f'{self.output_dir}/fig').mkdir(exist_ok=True, parents=True)
        pathlib.Path(f'{self.output_dir}/npy').mkdir(exist_ok=True, parents=True)

    def on_epoch_end(self, epoch, logs={}):
        decoded_imgs = self.model.predict(self.tests, verbose=2)[0]
        plt.title(f'epoch:{epoch:03}')
        plt.grid(False)
        plt.imshow(decoded_imgs, vmin=-1.6, vmax=1.6)
        plt.savefig(f'{self.output_dir}/fig/{epoch:06}.png')
        np.save(f'{self.output_dir}/npy/{epoch:06}.npy', decoded_imgs)