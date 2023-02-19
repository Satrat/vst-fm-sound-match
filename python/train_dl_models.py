import spiegelib as spgl
import numpy as np
import tensorflow as tf
import os
import json

def trainMLP(output_folder, epochs=100):
    trainFeatures = np.load(os.path.join(output_folder, 'train_features.npy'))
    trainParams = np.load(os.path.join(output_folder, 'train_patches.npy'))
    testFeatures = np.load(os.path.join(output_folder, 'test_features.npy'))
    testParams = np.load(os.path.join(output_folder, 'test_patches.npy'))

    # Flatten feature time slices
    trainFeatures = trainFeatures.reshape(trainFeatures.shape[0], -1)
    testFeatures = testFeatures.reshape(testFeatures.shape[0], -1)

    # Setup callbacks for trainings
    logger = spgl.estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Instantiate MLP Model
    mlp = spgl.estimator.MLP((trainFeatures.shape[-1],),
                                trainParams.shape[-1],
                                callbacks=[logger, earlyStopping])

    # Add training and validation data
    mlp.add_training_data(trainFeatures, trainParams)
    mlp.add_testing_data(testFeatures, testParams)
    mlp.model.summary()

    mlp.fit(epochs=epochs)
    mlp.save_model(os.path.join(output_folder, 'simple_fm_mlp.h5'))

    _, plot_data = logger.get_plotting_data()
    with open(os.path.join(output_folder, "mlp_logger.json"), "w") as outfile:
        json.dump(plot_data, outfile, indent=4)

def trainLSTM(output_folder, epochs=100):
    trainFeatures = np.load(os.path.join(output_folder, 'train_features.npy'))
    trainParams = np.load(os.path.join(output_folder, 'train_patches.npy'))
    testFeatures = np.load(os.path.join(output_folder, 'test_features.npy'))
    testParams = np.load(os.path.join(output_folder, 'test_patches.npy'))

    # Setup callbacks for trainings
    logger = spgl.estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    lstm = spgl.estimator.LSTM(trainFeatures.shape[-2:],
                                trainParams.shape[-1],
                                callbacks=[logger, earlyStopping])

    lstm.add_training_data(trainFeatures, trainParams)
    lstm.add_testing_data(testFeatures, testParams)
    lstm.model.summary()

    lstm.fit(epochs=epochs)

    lstm.save_model(os.path.join(output_folder, 'simple_fm_lstm.h5'))

    _, plot_data = logger.get_plotting_data()
    with open(os.path.join(output_folder, "lstm_logger.json"), "w") as outfile:
        json.dump(plot_data, outfile, indent=4)

def trainBiLSTM(output_folder, epochs=100, highway_layers=6):
    trainFeatures = np.load(os.path.join(output_folder, 'train_features.npy'))
    trainParams = np.load(os.path.join(output_folder, 'train_patches.npy'))
    testFeatures = np.load(os.path.join(output_folder, 'test_features.npy'))
    testParams = np.load(os.path.join(output_folder, 'test_patches.npy'))

    # Setup callbacks for trainings
    logger = spgl.estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    bi_lstm = spgl.estimator.HwyBLSTM(trainFeatures.shape[-2:],
                                    trainParams.shape[-1],
                                    callbacks=[logger, earlyStopping],
                                    lstm_size=512,
                                    highway_layers=highway_layers)

    bi_lstm.add_training_data(trainFeatures, trainParams)
    bi_lstm.add_testing_data(testFeatures, testParams)
    bi_lstm.model.summary()

    bi_lstm.fit(epochs=epochs)

    bi_lstm.save_model(os.path.join(output_folder, 'simple_fm_bi_lstm.h5'))
    
    _, plot_data = logger.get_plotting_data()
    with open(os.path.join(output_folder, "bi_lstm_logger.json"), "w") as outfile:
        json.dump(plot_data, outfile, indent=4)
