import pathlib, datetime
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

# my modules
from model import build_model
from mygenerator import two_input_generator
from plot_history import PlotHistory
from show_fig import show


def main():
    # define model
    model = build_model()
    print(model.summary())

    # compile
    lr = 0.0005
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # data generator
    train_datagen = two_input_generator('../patch_data/train_data.txt')
    valid_datagen = two_input_generator('../patch_data/validation_data.txt', val_mode=True)

    # callbacks
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    output = pathlib.Path(f'../output/{now}')
    output.mkdir(exist_ok=True, parents=True)
    cp = ModelCheckpoint(filepath = f'{output}/model.h5', monitor='val_loss',
            save_best_only=True, save_weights_only=False, verbose=0,  mode='auto')
    ph = PlotHistory(save_interval=1, dir_name=f'{output}', csv_output=True)
    ## learning rate
    def step_decay(epoch):
        return lr * 0.95 ** (epoch // 30)
    lr_decay = LearningRateScheduler(step_decay, verbose=1)
    cbs = [cp, ph, lr_decay, show(output)]

    # START training
    batch_size = 64
    model.fit_generator(
        generator=train_datagen.flow_from_directory(batch_size),
        steps_per_epoch=len(train_datagen.data_index) // batch_size,
        epochs=296,  # 100000 / (len(train_datagen.data_index) // batch_size)
        verbose=1,
        callbacks=cbs,
        validation_data=valid_datagen.flow_from_directory(batch_size),
        validation_steps=len(valid_datagen.data_index) // batch_size,
        max_queue_size=20
    )


if __name__ == "__main__":
    main()