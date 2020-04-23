import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np
import csv
import datetime, os
from keras.callbacks import Callback

# plot style
plt.style.use('ggplot') 

class PlotHistory(Callback):
    def __init__(self, save_interval=1, dir_name='./', csv_output=False, title=''):
        self.interval = save_interval
        self.dir_name = dir_name
        self.csv_output = csv_output
        self.title = title
        if csv_output:
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d_%H%M")
            if os.path.exists(f'./{dir_name}/history.csv'):
                os.rename(f'./{dir_name}/history.csv', f'./{dir_name}/history_{now}.csv')
            if os.path.exists(f'./{dir_name}/learning_curve.png'):
                os.rename(f'./{dir_name}/learning_curve.png', f'./{dir_name}/learning_curve.png')
        
    def on_train_begin(self, logs=None):
        self.history = {}
        self.history['loss'] = []
        self.history['acc'] = []
        self.do_validation = self.params['do_validation']
        if self.do_validation:
            self.history['val_loss'] = []
            self.history['val_acc'] = []

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(1)
        if self.do_validation:
            self.history['val_loss'].append(logs.get('val_loss'))
            self.history['val_acc'].append(1)
        if (epoch-1) % self.interval == 0:
            plot_history(history=self.history, dir_name=self.dir_name, csv_output=self.csv_output, title=self.title)

    def on_train_end(self, logs=None):
        plot_history(history=self.history, dir_name=self.dir_name, csv_output=self.csv_output, title=self.title)

def plot_history(history, begin_epoch=1, dir_name=None, csv_output=True, title='learning_curve'):
    # plot init settings
    begin_epoch -= 1
    val_exist = 'val_acc' in history.keys()
    plt.figure(figsize=(18, 7))
    plt.suptitle(title, fontsize=16)
    
    # plot accuracy settings
    plt.subplot(121)
    plt.title(f'model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    # plot accuracy
    plt.plot(list(range(begin_epoch+1, len(history['acc'][begin_epoch:])+1)), history['acc'][begin_epoch:])
    if val_exist:
        plt.plot(list(range(begin_epoch+1, len(history['val_acc'][begin_epoch:])+1)), history['val_acc'][begin_epoch:])
        plt.legend(['acc', 'val_acc'], loc='lower right')
    else:
        plt.legend(['acc'], loc='lower right')
    
    # plot loss settings
    plt.subplot(122)
    plt.title(f'model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    # plot loss
    plt.plot(list(range(begin_epoch+1, len(history['loss'][begin_epoch:])+1)), history['loss'][begin_epoch:])
    if val_exist:
        plt.plot(list(range(begin_epoch+1, len(history['val_loss'][begin_epoch:])+1)), history['val_loss'][begin_epoch:])
        plt.legend(['loss', 'val_loss'], loc='upper right')
    else:
        plt.legend(['loss'], loc='upper right')
    
    # show or save?
    if dir_name is None:
        plt.show()
    else:
        plt.savefig(f'{dir_name}/learning_curve.png')
        if csv_output:
            values = []
            for key in history.keys():
                values.append(history[key])
            values = np.array(values)
            with open(f'./{dir_name}/history.csv', 'w') as f_handle:
                writer = csv.writer(f_handle, lineterminator="\n")
                writer.writerows([history.keys()])  # header
                np.savetxt(f_handle, values.T, fmt="%.6f", delimiter=',')
    plt.close('all')
