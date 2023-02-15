from synth_config import configDexed,configWTSynth
from dataset_generation import initMFCCFeatures, generateDataset, generateEval
from train_dl_models import trainMLP, trainLSTM, trainBiLSTM
from sound_match_genetic import runGeneticAlgBasic, runGeneticAlgNSGA
from sound_match_dl import runMLP, runLSTM, runBiLSTM
from evaluation import evaluate
import os

synth_path = "/home/ubuntu/spiegelib_tests/WaveTableSynth/WaveTableSynth.so"
output_folder = "./runs/wt_mfcc_run2"
synth_state = os.path.join(output_folder, "WT_simple_params_with_lfo.json")
output_folder_eval = os.path.join(output_folder, "evaluation")

# dataset generation MFCC
configWTSynth(synth_path, synth_state)
features = initMFCCFeatures(num_mfccs=13, frame_size=2048, hop_size=1024)
generateDataset(synth_path, output_folder, synth_state, features, train_size=50000, test_size=10000, note_len=1.5, render_len=1.5)

# train DL models
trainMLP(output_folder, epochs=100)
trainLSTM(output_folder, epochs=100)
trainBiLSTM(output_folder, epochs=100)


# run deep learning sound matching algorithms
generateEval(synth_path, output_folder_eval, synth_state, features, num_samples=25, note_len=1.5, render_len=1.5)
runMLP(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.5, render_len=1.5)
runLSTM(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.5, render_len=1.5)
runBiLSTM(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.5, render_len=1.5)

# evaluate sound matching results for all models
evaluate(output_folder_eval, models = ['mlp', 'lstm', 'bi_lstm'])
