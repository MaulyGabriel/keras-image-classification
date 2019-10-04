from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import cv2
import imutils


class Cnn:

    def __init__(self):
        self.HEIGHT = 300
        self.WIDTH = 300
        self.BATCH_SIZE = 8
        self.EPOCHS = 10
        self.TOTAL_IMAGES = 600
        self.model = ResNet50(weight='imagenet', include_top=False, input_shape=(self.HEIGHT, self.WIDTH, 3))
        self.train = 'data'
        self.classes = ['movidius', 'raspberry']
        self.layers = [1024, 1024]
        self.dropout = 0.5
        self.adam = Adam(lr=0.00001)
        self.path_model = './checkpoints/' + 'ResNet50' + '_model_weights.h5'
        self.checkpoint = ModelCheckpoint(self.path_model, monitor=['acc'], verbose=1, mode='max')
        self.callbacks = [self.checkpoint]

    def generate_more_images(self):
        train_data = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True
        )

        train_generator = train_data.flow_from_directory(self.train, target_size=(self.HEIGHT, self.WIDTH),
                                                         batch_size=self.BATCH_SIZE)

        return train_generator

    def build_fine_tune_model(self, dropout, full_layers, num_classes):

        for layer in self.model.layers:
            layer.trainable = False

        x = self.model.ooutput
        x = Flatten()(x)

        for fc in full_layers:
            x = Dense(fc, activation='relu')(x)
            x = Dropout(dropout)(x)

        predictions = Dense(num_classes, activation='softmax')(x)

        fine_tune_model = Model(inputs=self.model.input, outputs=predictions)

        return fine_tune_model

    def complete_model(self):

        fine_tune_model = self.build_fine_tune_model(dropout=self.dropout, full_layers=self.layers,
                                                     num_classes=len(self.classes))

        return fine_tune_model

    @staticmethod
    def plot_training(history):

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')

        plt.show()

        plt.savefig('acc_vs_epochs.png')

    def train_model(self):

        model = self.complete_model()
        model.compile(self.adam, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit_generator(self.generate_more_images(), epochs=self.EPOCHS, workers=8,
                                      steps_per_epoch=self.TOTAL_IMAGES // self.BATCH_SIZE, shuffle=True,
                                      callbacks=self.callbacks)

        self.plot_training(history=history)

    def save_frame(self, path):
        video = cv2.VideoCapture(path)
        success, image = video.read()
        count = 0

        while success:
            image = imutils.resize(image, width=300, height=300)
            cv2.imwrite(self.train + "raspberry/raspberry%d.jpg" % count, image)
            success, image = video.read()
            print('Read a new frame: ', success)
            count += 1
