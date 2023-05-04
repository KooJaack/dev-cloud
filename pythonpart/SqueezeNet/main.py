import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, ReLU, concatenate, MaxPool2D, Dropout, AvgPool2D, Flatten


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "AugmentedAlzheimerDataset",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=[176, 208],
        batch_size=16,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "AugmentedAlzheimerDataset",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=[176, 208],
        batch_size=16,
    )

    class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
    train_ds.class_names = class_names
    val_ds.class_names = class_names

    NUM_CLASSES = len(class_names)

    def one_hot_label(image, label):
        print(f"before: {label[0]} ")
        label = tf.one_hot(label, NUM_CLASSES)
        print(f"after: {label[0]} \n")
        return image, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = train_ds.take(5000)
    val_ds = val_ds.take(1000)

    def fire_module(x, s1, e1, e3):
        s1x = Conv2D(s1, kernel_size=1, padding='same')(x)
        s1x = ReLU()(s1x)
        e1x = Conv2D(e1, kernel_size=1, padding='same')(s1x)
        e3x = Conv2D(e3, kernel_size=3, padding='same')(s1x)
        x = concatenate([e1x, e3x])
        x = ReLU()(x)
        return x

    def SqueezeNet(input_shape, nclasses):
        inpt = Input(input_shape)
        x = Conv2D(96,kernel_size=(7,7),strides=(2,2),padding='same')(inpt)
        x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
        x = fire_module(x, s1 = 16, e1 = 64, e3 = 64) #2
        x = fire_module(x, s1 = 16, e1 = 64, e3 = 64) #3
        x = fire_module(x, s1 = 32, e1 = 128, e3 = 128) #4
        x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
        x = fire_module(x, s1 = 32, e1 = 128, e3 = 128) #5
        x = fire_module(x, s1 = 48, e1 = 192, e3 = 192) #6
        x = fire_module(x, s1 = 48, e1 = 192, e3 = 192) #7
        x = fire_module(x, s1 = 64, e1 = 256, e3 = 256) #8
        x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
        x = fire_module(x, s1 = 64, e1 = 256, e3 = 256) #9
        x = Dropout(0.5)(x)
        x = Conv2D(nclasses,kernel_size = 1)(x)
        output = AvgPool2D(pool_size=[x.shape.dims[1], x.shape.dims[2]])(x)
        output = Flatten()(output)
        model = Model(inpt, output)
        return model


    IMAGE_SIZE = (176, 208, 3)
    model = SqueezeNet(IMAGE_SIZE, NUM_CLASSES)
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.CategoricalAccuracy(name='acc')]
    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(), metrics=METRICS)
    fitmodel = model.fit(train_ds, validation_data=val_ds, epochs=15)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
