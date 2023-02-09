import spiegelib as spgl
import numpy as np

def evaluate():
    # Load the sound targets used for sound matching
    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')

    # Load all the estimations of the sound targets made by each estimator
    estimations = [spgl.AudioBuffer.load_folder('./evaluation/mlp'),
                spgl.AudioBuffer.load_folder('./evaluation/lstm'),
                spgl.AudioBuffer.load_folder('./evaluation/bi_lstm'),
                spgl.AudioBuffer.load_folder('./evaluation/cnn'),
                spgl.AudioBuffer.load_folder('./evaluation/ga'),
                spgl.AudioBuffer.load_folder('./evaluation/nsga')]

    # Evaluate the results and save to JSON file
    evaluation = spgl.evaluation.MFCCEval(targets, estimations)
    evaluation.evaluate()
    evaluation.save_stats_json('./evaluation/evaluation_stats.json')
    evaluation.save_scores_json('./evaluation/evaluation_scores.json')