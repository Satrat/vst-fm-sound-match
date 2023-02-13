import spiegelib as spgl

def generateDatasetMFCC(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    features = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
    generator = spgl.DatasetGenerator(synth, features,
                                  output_folder="./data_simple_FM_mfcc",
                                  scale=True)

    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_scaler('data_scaler.pkl')

    return features

def generateDatasetSTFT(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    # Magnitude STFT ouptut feature extraction
    features = spgl.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)

    # Setup generator and create dataset
    generator = spgl.DatasetGenerator(synth, features, output_folder="./data_simple_FM_stft", scale=True)
    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_scaler('data_scaler.pkl')

    return features

def generateEval(synth_path, features, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    eval_generator = spgl.DatasetGenerator(synth, features,
                                        output_folder='./evaluation',
                                        save_audio=True)
    eval_generator.generate(25)

