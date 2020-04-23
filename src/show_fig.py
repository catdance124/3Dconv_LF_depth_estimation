import keras
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
class show(keras.callbacks.Callback):
    def __init__(self, tests_h, tests_v,**kwargs):
        super().__init__(**kwargs)
        self.tests = [np.load('../patch_data/pillows/full_h.npy')[np.newaxis],
                        np.load('../patch_data/pillows/full_v.npy')[np.newaxis]]
        self.count = 0

    def on_batch_end(self, batch, logs={}):
        if (batch+1) % 150 == 0:
            decoded_imgs = self.model.predict(self.tests, verbose=2)
            print('\n\n^^^^^^^^^^^^^^^^\n')
            print(np.load('../patch_data/pillows/full_disp.npy')+10)
            print(decoded_imgs[0])
            plt.imshow(decoded_imgs[0])
            plt.savefig(f'../output/fig/figure_{self.count:06}.png')
            self.count+=1