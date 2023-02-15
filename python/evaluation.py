import spiegelib as spgl
import numpy as np
import os

def evaluate(eval_folder, models=['mlp', 'lstm', 'bi_lstm', 'ga', 'nsga']):
    # Load the sound targets used for sound matching
    targets = spgl.AudioBuffer.load_folder(os.path.join(eval_folder, 'audio'))

    # Load all the estimations of the sound targets made by each estimator
    estimations = [spgl.AudioBuffer.load_folder(os.path.join(eval_folder, model_name)) for model_name in models]

    # Evaluate the results and save to JSON file
    evaluation = spgl.evaluation.MFCCEval(targets, estimations)
    evaluation.evaluate()
    evaluation.save_stats_json(os.path.join(eval_folder, 'evaluation_stats.json'))
    evaluation.save_scores_json(os.path.join(eval_folder, 'evaluation_scores.json'))
