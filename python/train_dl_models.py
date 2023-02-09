import spiegelib as spgl
import numpy as np
import tensorflow as tf

def trainMLP():
    trainFeatures = np.load('./data_simple_FM_mfcc/train_features.npy')
    trainParams = np.load('./data_simple_FM_mfcc/train_patches.npy')
    testFeatures = np.load('./data_simple_FM_mfcc/test_features.npy')
    testParams = np.load('./data_simple_FM_mfcc/test_patches.npy')

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

    mlp.fit(epochs=100)
    mlp.save_model('./saved_models/simple_fm_mlp.h5')
    logger.plot()

def trainLSTM():
    trainFeatures = np.load('./data_simple_FM_mfcc/train_features.npy')
    trainParams = np.load('./data_simple_FM_mfcc/train_patches.npy')
    testFeatures = np.load('./data_simple_FM_mfcc/test_features.npy')
    testParams = np.load('./data_simple_FM_mfcc/test_patches.npy')

    # Setup callbacks for trainings
    logger = spgl.estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    lstm = spgl.estimator.LSTM(trainFeatures.shape[-2:],
                                trainParams.shape[-1],
                                callbacks=[logger, earlyStopping])

    lstm.add_training_data(trainFeatures, trainParams)
    lstm.add_testing_data(testFeatures, testParams)
    lstm.model.summary()

    lstm.fit(epochs=100)

    lstm.save_model('./saved_models/simple_fm_lstm.h5')
    logger.plot()

def trainLSTMPlusPlus():

    trainFeatures = np.load('./data_simple_FM_mfcc/train_features.npy')
    trainParams = np.load('./data_simple_FM_mfcc/train_patches.npy')
    testFeatures = np.load('./data_simple_FM_mfcc/test_features.npy')
    testParams = np.load('./data_simple_FM_mfcc/test_patches.npy')

    # Flatten feature time slices
    trainFeatures = trainFeatures.reshape(trainFeatures.shape[0], -1)
    testFeatures = testFeatures.reshape(testFeatures.shape[0], -1)

    # Setup callbacks for trainings
    logger = spgl.estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    bi_lstm = spgl.estimator.HwyBLSTM(trainFeatures.shape[-2:],
                                    trainParams.shape[-1],
                                    callbacks=[logger, earlyStopping],
                                    highway_layers=6)

    bi_lstm.add_training_data(trainFeatures, trainParams)
    bi_lstm.add_testing_data(testFeatures, testParams)
    bi_lstm.model.summary()

    bi_lstm.fit(epochs=100)

    bi_lstm.save_model('./saved_models/simple_fm_bi_lstm.h5')
    logger.plot()