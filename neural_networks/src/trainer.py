import copy

import tensorflow as tf
import flammkuchen as fl
import numpy as np
from .dataloader import DataLoader
import matplotlib.pyplot as plt
import os
import time
from sklearn.utils.class_weight import compute_class_weight
from src.models import get_resnet18, get_resnet34, get_efficientnetb0, get_mobilenetv2, get_rnn_amoh, get_rnn_basic
from src.lr_scheduler import exp_scheduler


class Trainer:
    def __init__(self, params, nb_classes, batch_size, nb_epochs):
        self.params = copy.deepcopy(params)
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        data_loader = DataLoader(params=params, nb_classes=nb_classes)

        (self.X_train, self.Y_train), (self.X_val, self.Y_val) = data_loader.get_train_val_data()

        if self.X_train is None or self.X_val is None or self.Y_val is None or self.Y_train is None:
            raise ValueError('No data created.')

    def train(self, architecture,
              results_dir=None,
              foldername=None,
              evaluate=None,
              class_weights=False,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy()):
        """
        Trains the given architecture on the previous loaded mel-spectrograms.
        Args:
            architecture:  Name of the architecture as string, e.g. 'ResNet18' or 'EfficientNetB0'.
            results_dir: Directory where to save the results. If 'None', the results won't be saved.
            foldername: Name of the folder that shall be created.
            evaluate: [X_test, Y_test] for evaluating the trained model. If 'None', the model won't be evaluated.
            class_weights: Class weights for training the model. If 'None', no class weights will be used.
            optimizer: Optimizer for training the model.
            loss: Loss function for training the model.
        Returns:
        The trained model, the complete training history, validation accuracy and test accuracy.
        """

        # load model and compile it
        model, scheduler = self.__get_model(architecture, self.nb_classes, self.params)
        scheduler = exp_scheduler
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=["accuracy"])

        class_weight = None
        if class_weights:
            class_weight = self.__get_class_weights()

        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
        callbacks = [lr_callback]

        if results_dir is not None:
            # join full path
            results_dir = os.path.join(results_dir, self.params.signal_type + '/', architecture)

            # create directory for saving the results
            cp_dir, new_folder_name = self.__create_saving_directory(results_dir, foldername)

            # create checkpoint callback
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_dir + "model.hdf5",  # "model_epoch{epoch:02d}.hdf5" "{val_loss:.2f}
                                                             monitor='val_accuracy',
                                                             mode='max',
                                                             verbose=True,
                                                             save_weights_only=True,
                                                             save_best_only=True)
            callbacks = [lr_callback, cp_callback]

        print("Start training")
        if architecture == 'RNN_Amoh' or architecture == 'RNN_Basic':
            history = model.fit(self.X_train, self.Y_train,
                                shuffle=True,
                                epochs=self.nb_epochs,
                                batch_size=self.batch_size,
                                callbacks=callbacks,
                                validation_data=(self.X_val, self.Y_val),
                                class_weight=class_weight)
        else:  # train CNN
            history = model.fit(self.X_train[..., None], self.Y_train,
                                shuffle=True,
                                epochs=self.nb_epochs,
                                batch_size=self.batch_size,
                                callbacks=callbacks,
                                validation_data=(self.X_val[..., None], self.Y_val),
                                class_weight=class_weight)

        val_acc, test_acc = None, None
        if evaluate is not None and results_dir is not None:
            model.load_weights(cp_dir + "model.hdf5")
            if architecture == 'RNN_Amoh' or architecture == 'RNN_Basic':
                _, val_acc = model.evaluate(self.X_val, self.Y_val, batch_size=self.batch_size)
                _, test_acc = model.evaluate(evaluate[0], evaluate[1], batch_size=self.batch_size)
            else:
                _, val_acc = model.evaluate(self.X_val[..., None], self.Y_val, batch_size=self.batch_size)
                _, test_acc = model.evaluate(evaluate[0][..., None], evaluate[1], batch_size=self.batch_size)

        if results_dir is not None:
            path = os.path.join(results_dir, new_folder_name)
            self.__plot_history(history, path, self.nb_epochs)
            self.__save_history(history, path, architecture)

        return model, history, val_acc, test_acc

    @staticmethod
    def __get_model(architecture, nb_classes, params):
        """
        Get the requested TensorFlow model.
        Args:
            architecture: Name of the requested architecture.
            nb_classes: Number of output classes.
            params: Parameter object containing several information.

        Returns:
        The requested TensorFlow architecture as TensorFlow model.
        """
        if type(architecture) == str:
            if architecture == 'ResNet18':
                return get_resnet18(nb_classes, params), resnet18_scheduler
            elif architecture == 'ResNet34':
                return get_resnet34(nb_classes, params), resnet34_scheduler
            elif architecture == 'EfficientNetB0':
                return get_efficientnetb0(nb_classes, params), efn_scheduler
            elif architecture == 'MobileNetV2':
                return get_mobilenetv2(nb_classes, params), mnet_scheduler
            elif architecture == 'RNN_Amoh':
                return get_rnn_amoh(nb_classes, params), standard_scheduler
            elif architecture == 'RNN_Basic':
                return get_rnn_basic(nb_classes, params), standard_scheduler
            else:
                raise ValueError("The given architecture is not available.")
        else:
            return architecture, ga_scheduler  # architecture is a tf-model already in this case

    def __get_class_weights(self):
        """
        Estimate class weights for unbalanced datasets.
        Returns:
        Array with class_weight_vect[i] the weight for i-th class.
        """
        y_train = []
        for y in self.Y_train:
            y_train.append(int(np.where(y == 1)[0]) + 1)

        y_train = np.array(y_train)

        class_weights = compute_class_weight(class_weight='balanced', classes=[1, 2, 3, 4], y=y_train)
        class_weights = {0: class_weights[0],
                         1: class_weights[1],
                         2: class_weights[2],
                         3: class_weights[3]}
        print(f"Using class weights: {class_weights}")
        return class_weights

    def __create_saving_directory(self, results_dir, foldername=None):
        """
        Creates a new directory for saving checkpoints, history-plots and params_used.json.
        Args:
            results_dir: Directory where to create the results folder.
            foldername: Name of the folder that shall be created.

        Returns:
        Directory for saving model checkpoints and the new folders name.
        """
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # new folder will just be named after the timestamp if the method argument is 'None'
        if foldername is None:
            new_folder_name = time.strftime("%Y%m%d-%H%M%S")
        else:
            new_folder_name = foldername
        dir_ = results_dir + '/' + new_folder_name
        os.makedirs(dir_)

        # save params in newly created folder
        self.params.save(dir_ + '/params_used.json')

        # create directory for saving checkpoints
        cp_dir = dir_ + '/Checkpoints/'
        os.makedirs(cp_dir)

        return cp_dir, new_folder_name

    @staticmethod
    def __plot_history(history, save_dir, nb_epochs):
        """
        Plots and saves the training history.
        Args:
            history: TensorFlow training history.
            save_dir: Directory where to save the history plots.
            nb_epochs: Number of epochs during training for plotting the history.
        """
        x = np.arange(nb_epochs)
        plt.plot(x, history.history['loss'])
        plt.plot(x, history.history['val_loss'])
        plt.legend(['Train loss', 'Validation loss'], loc='upper right')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(save_dir + "\\loss.png")

        plt.cla()

        plt.plot(x, history.history['accuracy'])
        plt.plot(x, history.history['val_accuracy'])
        plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig(save_dir + "\\accuracy.png")

    @staticmethod
    def __save_history(history, save_dir, architecture):
        """
        Saving TensorFlow training history using 'flammkuchen'.
        Args:
            history: TensorFlow training history.
            save_dir: Directory where to save the history plots.
            architecture: Name of the architecture used.
        """
        history = dict(history.history, architecture=architecture)
        fl.save(os.path.join(save_dir, "trainHistoryDict.history"), history)
