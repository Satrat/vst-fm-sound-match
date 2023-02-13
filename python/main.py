from synth_config import configDexed,configWTSynth
from dataset_generation import generateDatasetMFCC, generateDatasetSTFT, generateEval
from train_dl_models import trainMLP, trainLSTM, trainLSTMPlusPlus
from sound_match_genetic import runGeneticAlgBasic, runGeneticAlgNSGA
from sound_match_dl import runMLP, runLSTM, runLSTMPlusPlus
from evaluation import evaluate

synth_path = "/Users/sadkins/Documents/infinite_album/dev/open_source/spiegelib_test/camomile_test/WaveTableSynth/WaveTableSynth.vst"
synth_state = "./synth_params_camomile/wt_synth_simple.json"

configWTSynth(synth_path, synth_state)
features = generateDatasetMFCC(synth_path, synth_state=synth_state, train_size=50000, test_size=10000)
generateEval(synth_path, features, synth_state=synth_state)

# dataset generation MFCC
configDexed(synth_path, synth_state=synth_state)
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
