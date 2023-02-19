import spiegelib as spgl
from pydub import AudioSegment
import os

def runGeneticAlgBasic(synth_path, model_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.0, render_len=1.0, pop_size=300,gen_size=100):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    # MFCC features
    ga_extractor = spgl.features.MFCC(num_mfccs=num_mfccs, hop_size=hop_size)

    # Basic Genetic Algorithm estimator
    ga = spgl.estimator.BasicGA(synth, ga_extractor, pop_size=pop_size, ngen=gen_size)

    # Sound matching helper class
    ga_matcher = spgl.SoundMatch(synth, ga)

    input_folder_audio = os.path.join(output_folder_eval, 'audio')
    targets = spgl.AudioBuffer.load_folder(os.path.join(output_folder_eval, 'audio'))
    output_folder_pred = os.path.join(output_folder_eval, 'ga')
    output_folder_concat = os.path.join(output_folder_eval, 'ga_concat')
    if not os.path.exists(output_folder_concat):
        os.mkdir(output_folder_concat)
    for i in range(len(targets)):
        target_path = os.path.join(input_folder_audio, 'output_%s.wav' % i)
        est_path = os.path.join(output_folder_pred, 'ga_prediction_%s.wav' % i)
        concat_path = os.path.join(output_folder_concat, 'concat_ga_prediction_%s.wav' % i)
        audio = ga_matcher.match(targets[i])
        audio.save(est_path)
        estimation = AudioSegment.from_wav(est_path)
        target = AudioSegment.from_wav(target_path)
        concat = target + estimation
        concat.export(concat_path, format="wav")


def runGeneticAlgNSGA(synth_path, model_folder, output_folder_eval, synth_state, num_mfccs=13, hop_size=1024, note_len=1.0, render_len=1.0, pop_size=300,gen_size=100):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=note_len, render_length_secs=render_len)
    synth.load_state(synth_state)

    # Feature extractors for Multi-Objective GA
    nsga_extractors = [spgl.features.MFCC(num_mfccs=num_mfccs, hop_size=hop_size),
                    spgl.features.SpectralSummarized(hop_size=hop_size),
                    spgl.features.FFT(output='magnitude')]

    # NSGA3 Multi-Objective Genetic Algorithm
    nsga = spgl.estimator.NSGA3(synth, nsga_extractors, pop_size=pop_size, ngen=gen_size)

    # Sound matching helper class
    nsga_matcher = spgl.SoundMatch(synth, nsga)

    input_folder_audio = os.path.join(output_folder_eval, 'audio')
    targets = spgl.AudioBuffer.load_folder(os.path.join(output_folder_eval, 'audio'))
    output_folder_pred = os.path.join(output_folder_eval, 'nsga')
    output_folder_concat = os.path.join(output_folder_eval, 'nsga_concat')
    if not os.path.exists(output_folder_concat):
        os.mkdir(output_folder_concat)
    for i in range(len(targets)):
        target_path = os.path.join(input_folder_audio, 'output_%s.wav' % i)
        est_path = os.path.join(output_folder_pred, 'nsga_prediction_%s.wav' % i)
        concat_path = os.path.join(output_folder_concat, 'concat_nsga_prediction_%s.wav' % i)
        audio = nsga_matcher.match(targets[i])
        audio.save(est_path)
        estimation = AudioSegment.from_wav(est_path)
        target = AudioSegment.from_wav(target_path)
        concat = target + estimation
        concat.export(concat_path, format="wav")
