from synth_config import configDexed,configWTSynth
from dataset_generation import initMFCCFeatures, generateDataset, generateEval
from train_dl_models import trainMLP, trainLSTM, trainBiLSTM
from sound_match_genetic import runGeneticAlgBasic, runGeneticAlgNSGA
from sound_match_dl import runMLP, runLSTM, runBiLSTM
from evaluation import evaluate
import os

# configurations
NOTE_LEN = 1.5
RENDER_LEN = 1.5
MFCCS = 13
FRAME = 1024
HOP = 512
EVAL_SAMPLES = 30
TRAIN_SIZE = 30000
TEST_SIZE = 8000
EPOCHS = 100
POP_SIZE = 300
GEN_SIZE = 100
OUTPUT_FOLDER_NAME = "wt_mfcc_run4"

# set up paths
synth_path = "/home/ubuntu/spiegelib_tests/WaveTableSynth/WaveTableSynth.so"
output_folder = os.path.join("./runs", OUTPUT_FOLDER_NAME)
synth_state = os.path.join(output_folder, "modified_param_space.json")
output_folder_eval = os.path.join(output_folder, "evaluation")

# dataset generation MFCC
configWTSynth(synth_path, synth_state)
features = initMFCCFeatures(num_mfccs=MFCCS, frame_size=FRAME, hop_size=HOP)
generateDataset(synth_path, output_folder, synth_state, features, train_size=TRAIN_SIZE, test_size=TEST_SIZE, note_len=NOTE_LEN, render_len=RENDER_LEN)

# train DL models
#trainMLP(output_folder, epochs=EPOCHS)
#trainLSTM(output_folder, epochs=EPOCHS)
trainBiLSTM(output_folder, epochs=EPOCHS)


# run deep learning sound matching algorithms
generateEval(synth_path, output_folder_eval, synth_state, features, num_samples=EVAL_SAMPLES, note_len=NOTE_LEN, render_len=RENDER_LEN)
#runMLP(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=MFCCS, hop_size=HOP, note_len=NOTE_LEN, render_len=RENDER_LEN)
#runLSTM(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=MFCCS, hop_size=HOP, note_len=NOTE_LEN, render_len=RENDER_LEN)
runBiLSTM(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=MFCCS, hop_size=HOP, note_len=NOTE_LEN, render_len=RENDER_LEN)

# run genetic algorithms
runGeneticAlgBasic(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=MFCCS, hop_size=HOP, note_len=NOTE_LEN, render_len=RENDER_LEN, pop_size=POP_SIZE, gen_size=GEN_SIZE)
runGeneticAlgNSGA(synth_path, output_folder, output_folder_eval, synth_state, num_mfccs=MFCCS, hop_size=HOP, note_len=NOTE_LEN, render_len=RENDER_LEN, pop_size=POP_SIZE, gen_size=GEN_SIZE)


# evaluate sound matching results for all models
evaluate(output_folder_eval, models = ['ga', 'nsga', 'bi_lstm'])
