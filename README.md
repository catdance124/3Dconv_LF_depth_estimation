# 3Dconv_LF_depth_estimation
implementation (using keras) of ( Faluvégi, Ágota, et al. "A 3D Convolutional Neural Network for Light Field Depth Estimation." 2019 International Conference on 3D Immersion (IC3D). IEEE, 2019.)

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
output/YYYY-MM-DD_HHmm/fig/figure_*.png  
![](predicts.gif)

## model architecture
![](model_plot.png)