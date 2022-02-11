from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
print('talmo_front_test1_div4')
# set image generators
path = '/home/a623255/PycharmProjects/Practice/Project_talmo/talmo4divde/'
train_dir = path+'train'
test_dir = path+'test'
validation_dir = path+'val'

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20, shear_range=0.1,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='categorical')

# model definition
input_shape = [128, 128, 3]  # as a shape of image


def build_model():
    model=models.Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='sigmoid'))

    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

import time

starttime = time.time();
num_epochs = 100
# num_epochs = 100 # Q2
model = build_model()
history = model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=261, validation_data=validation_generator, validation_steps=1.25)

model.save('project_div4_test1.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_loss:', train_loss)
print('train_acc:', train_acc)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time() - starttime)

# visualization


def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


plot_loss(history)
plt.savefig('talmo_front_testdiv4_loss.png')
plt.clf()
plot_acc(history)
plt.savefig('talmo_front_testdiv4_acc.png')