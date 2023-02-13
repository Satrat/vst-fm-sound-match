import spiegelib as spgl
import numpy as np
import matplotlib.pyplot as plt

def runMLP(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    mlp = spgl.estimator.TFEstimatorBase.load('./saved_models/simple_fm_mlp.h5')

    # MLP feature extractor with a modifying function that flattens the time slice arrays at the end of the feature
    # extraction pipeline
    mlp_extractor = spgl.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024, scale=True)
    mlp_extractor.load_scaler('./data_simple_FM_mfcc/data_scaler.pkl')
    mlp_extractor.add_modifier(lambda data : data.flatten(), type='output')
    mlp_matcher = spgl.SoundMatch(synth, mlp, mlp_extractor)

    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')
    for i in range(len(targets)):
        audio = mlp_matcher.match(targets[i])
        audio.save('./evaluation/mlp/mlp_prediction_%s.wav' % i)
        

def runLSTM(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    lstm = spgl.estimator.TFEstimatorBase.load('./saved_models/simple_fm_lstm.h5')

    # LSTM & LSTM++ feature extractor -- time series of MFCC frames
    lstm_extractor = spgl.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024, scale=True)
    lstm_extractor.load_scaler('./data_simple_FM_mfcc/data_scaler.pkl')
    lstm_matcher = spgl.SoundMatch(synth, lstm, lstm_extractor)

    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')
    for i in range(len(targets)):
        audio = lstm_matcher.match(targets[i])
        audio.save('./evaluation/lstm/lstm_prediction_%s.wav' % i)


def runLSTMPlusPlus(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    bi_lstm = spgl.estimator.TFEstimatorBase.load('./saved_models/simple_fm_bi_lstm.h5')

    # LSTM & LSTM++ feature extractor -- time series of MFCC frames
    lstm_extractor = spgl.features.MFCC(num_mfccs=13, time_major=True, hop_size=1024, scale=True)
    lstm_extractor.load_scaler('./data_simple_FM_mfcc/data_scaler.pkl')
    bi_lstm_matcher = spgl.SoundMatch(synth, bi_lstm, lstm_extractor)

    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')
    for i in range(len(targets)):
        audio = bi_lstm_matcher.match(targets[i])
        audio.save('./evaluation/bi_lstm/bi_lstm_prediction_%s.wav' % i)
