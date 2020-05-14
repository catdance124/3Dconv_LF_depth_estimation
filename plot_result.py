import pandas as pd
import numpy as np
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plot style
plt.style.use('ggplot')
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


output_dir = './output/2020-05-12_2238/'
df = pd.read_csv(f'{output_dir}/history.csv')
df = df[['loss', 'val_loss']]

disp = np.load(f'./patch_data/town/full_disp.npy')

for i in range(296):
    pred = np.load(f'{output_dir}/npy/{i:06}.npy')-5
    
    plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2,3)

    plt.subplot(gs[0,:])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(df)
    plt.ylim(0, 0.15)
    plt.axvline(i, color = "navy")

    plt.subplot(gs[1:,0])
    plt.title('GT')
    plt.grid(False)
    plt.imshow(disp, vmax=1.6, vmin=-1.6)
    plt.colorbar(shrink=0.8)

    plt.subplot(gs[1:,1])
    plt.title('pred')
    plt.grid(False)
    plt.imshow(pred, vmax=1.6, vmin=-1.6)
    plt.colorbar(shrink=0.8)

    plt.subplot(gs[1:,2])
    plt.title('diff (pred-GT)')
    plt.grid(False)
    plt.imshow(pred-disp, cmap='bwr', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar(shrink=0.8)

    pathlib.Path(f'{output_dir}/result').mkdir(exist_ok=True, parents=True)
    plt.savefig(f'{output_dir}/result/{i:06}.png')
    plt.close()