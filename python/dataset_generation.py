import spiegelib as spgl

def initMFCCFeatures(num_mfccs=13, frame_size=2048, hop_size=1024):
    return spgl.features.MFCC(num_mfccs=num_mfccs, frame_size=frame_size, hop_size=hop_size, time_major=True)

def initSTFTFeatures(fft_size=2048, hop_size=1024):
    return spgl.features.STFT(fft_size=fft_size, hop_size=hop_size, output='magnitude', time_major=True)

def generateDataset(synth_path, output_folder, synth_state, features, train_size=10000, test_size=1000, note_len=1.0, render_len=1.0):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    generator = spgl.DatasetGenerator(synth, features,
                                  output_folder=output_folder,
                                  scale=True)

    generator.generate(train_size, file_prefix="train_")
    generator.generate(test_size, file_prefix="test_")
    generator.save_scaler('data_scaler.pkl')

def generateEval(synth_path, output_folder, synth_state, features, num_samples=25, note_len=1.0, render_len=1.0):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    eval_generator = spgl.DatasetGenerator(synth, features,
                                        output_folder=output_folder,
                                        save_audio=True)
    eval_generator.generate(num_samples)

