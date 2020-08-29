# 3Dconv_LF_depth_estimation
implementation (using keras) of ( Faluvégi, Ágota, et al. "A 3D Convolutional Neural Network for Light Field Depth Estimation." 2019 International Conference on 3D Immersion (IC3D). IEEE, 2019.)

## Implementation env
- Python 3.6.5
- Keras==2.2.2
- tensorflow-gpu==1.10.0

## To run
Download light field dataset (from https://lightfield-analysis.uni-konstanz.de/).  

Please set up the file structure as follows.  
```
3Dconv_LF_depth_estimation/
  ┣━━ src/    ...    source codes
  ┣━━ output/    ...    dir for output (this will be created later automatically created.)
  ┣━━ patch_data/    ...    dir for patch data (the data will be created later.)
  ┃     ┣━━ train_data.txt        ...    scenes to use for training
  ┃     ┣━━ validation_data.txt   ...    scenes to use for validation
  ┃     ┗━━ test_data.txt         ...    scenes to use for test
  ┣━━ full_data/    ...    downloaded dataset
  ┃     ┣━━ additional/
  ┃     ┣━━ stratified/
  ┃     ┣━━ test/
  ┃     ┗━━ training/
  ┣━━ plot_result.py    ...    the script to create the figure below
  ┗━━ README.md    ...    this document
```

Create patch dataset(The first time only.)  
```
cd src
python create_dataset.py
```

Start training
```
cd src
python train.py
```

## result
The predicts for each epoch are placed here.  
output/YYYY-MM-DD_HHmm/fig/{epoch}.png  
  
Each frame of the following figure was created by [plot_result.py](./plot_result.py).  
I used [Giam](http://furumizo.net/tsu/giamd.htm) to connect each frame and created the following figure.  
```
# after rewrite the output_dir variable in ./plot_result.py
python ./plot_result.py
# save to ./output/YYYY-MM-DD_HHmm/result/{epoch}.png  
```

![](result.gif)

## model architecture
![](model_plot.png)