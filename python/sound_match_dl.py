import spiegelib as spgl
import numpy as np
import matplotlib.pyplot as plt
import os

def runMLP(synth_path, model_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.0, render_len=1.0):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    mlp = spgl.estimator.TFEstimatorBase.load(os.path.join(model_folder, 'simple_fm_mlp.h5'))

    # MLP feature extractor with a modifying function that flattens the time slice arrays at the end of the feature
    # extraction pipeline
    mlp_extractor = spgl.features.MFCC(num_mfccs=num_mfccs, time_major=True, hop_size=hop_size, scale=True)
    mlp_extractor.load_scaler(os.path.join(model_folder, 'data_scaler.pkl'))
    mlp_extractor.add_modifier(lambda data : data.flatten(), type='output')
    mlp_matcher = spgl.SoundMatch(synth, mlp, mlp_extractor)

    targets = spgl.AudioBuffer.load_folder(os.path.join(output_folder_eval, 'audio'))
    output_folder_pred = os.path.join(output_folder_eval, 'mlp')
    for i in range(len(targets)):
        audio = mlp_matcher.match(targets[i])
        audio.save(os.path.join(output_folder_pred, 'mlp_prediction_%s.wav' % i))
        

def runLSTM(synth_path, model_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.0, render_len=1.0):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    lstm = spgl.estimator.TFEstimatorBase.load(os.path.join(model_folder, 'simple_fm_lstm.h5'))

    # LSTM & LSTM++ feature extractor -- time series of MFCC frames
    lstm_extractor = spgl.features.MFCC(num_mfccs=num_mfccs, time_major=True, hop_size=hop_size, scale=True)
    lstm_extractor.load_scaler(os.path.join(model_folder, 'data_scaler.pkl'))
    lstm_matcher = spgl.SoundMatch(synth, lstm, lstm_extractor)

    targets = spgl.AudioBuffer.load_folder(os.path.join(output_folder_eval, 'audio'))
    output_folder_pred = os.path.join(output_folder_eval, 'lstm')
    for i in range(len(targets)):
        audio = lstm_matcher.match(targets[i])
        audio.save(os.path.join(output_folder_pred, 'lstm_prediction_%s.wav' % i))


def runBiLSTM(synth_path, model_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.0, render_len=1.0):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    bi_lstm = spgl.estimator.TFEstimatorBase.load(os.path.join(model_folder, 'simple_fm_bi_lstm.h5'))

    # LSTM & LSTM++ feature extractor -- time series of MFCC frames
    lstm_extractor = spgl.features.MFCC(num_mfccs=num_mfccs, time_major=True, hop_size=hop_size, scale=True)
    lstm_extractor.load_scaler(os.path.join(model_folder, 'data_scaler.pkl'))
    bi_lstm_matcher = spgl.SoundMatch(synth, bi_lstm, lstm_extractor)

    targets = spgl.AudioBuffer.load_folder(os.path.join(output_folder_eval, 'audio'))
    output_folder_pred = os.path.join(output_folder_eval, 'bi_lstm')
    for i in range(len(targets)):
        audio = bi_lstm_matcher.match(targets[i])
        audio.save(os.path.join(output_folder_pred, 'bi_lstm_prediction_%s.wav' % i))
