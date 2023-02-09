from synth_config import configSynth
from dataset_generation import generateDatasetMFCC, generateDatasetSTFT, generateEval
from train_dl_models import trainMLP, trainLSTM, trainLSTMPlusPlus
from sound_match_genetic import runGeneticAlgBasic, runGeneticAlgNSGA
from sound_match_dl import runMLP, runLSTM, runLSTMPlusPlus
from evaluation import evaluate

synth_path = "/Library/Audio/Plug-Ins/Components/Dexed.component"
synth_state = "./synth_params/dexed_simple_fm.json"

# dataset generation MFCC
configSynth(synth_path, synth_state=synth_state)
features = generateDatasetMFCC(synth_path, synth_state=synth_state)
generateEval(synth_path, features, synth_state=synth_state)

# train DL models
trainMLP()
trainLSTM()
trainLSTMPlusPlus()

# run genetic sound matching algorithms
runGeneticAlgBasic(synth_path, synth_state=synth_state)
runGeneticAlgNSGA(synth_path, synth_state=synth_state)

# run deep learning sound matching algorithms
runMLP(synth_path, synth_state=synth_state)
runLSTM(synth_path, synth_state=synth_state)
runLSTMPlusPlus(synth_path, synth_state=synth_state)

# evaluate sound matching results for all models
evaluate()
