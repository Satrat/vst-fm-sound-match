import spiegelib as spgl

def runGeneticAlgBasic(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    # MFCC features
    ga_extractor = spgl.features.MFCC(num_mfccs=13, hop_size=1024)

    # Basic Genetic Algorithm estimator
    ga = spgl.estimator.BasicGA(synth, ga_extractor, pop_size=300, ngen=100)

    # Sound matching helper class
    ga_matcher = spgl.SoundMatch(synth, ga)

    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')

    for i in range(len(targets)):
        audio = ga_matcher.match(targets[i])
        audio.save('./evaluation/ga/ga_predicition_%s.wav' % i)


def runGeneticAlgNSGA(synth_path, synth_state='./synth_params/dexed_simple_fm.json'):
    synth = spgl.synth.SynthVST(synth_path,
                                note_length_secs=1.0, render_length_secs=1.0)
    synth.load_state(synth_state)

    # Feature extractors for Multi-Objective GA
    nsga_extractors = [spgl.features.MFCC(num_mfccs=13, hop_size=1024),
                    spgl.features.SpectralSummarized(hop_size=1024),
                    spgl.features.FFT(output='magnitude')]

    # NSGA3 Multi-Objective Genetic Algorithm
    nsga = spgl.estimator.NSGA3(synth, nsga_extractors, pop_size=300, ngen=100)

    # Sound matching helper class
    nsga_matcher = spgl.SoundMatch(synth, nsga)

    targets = spgl.AudioBuffer.load_folder('./evaluation/audio')

    for i in range(len(targets)):

        audio = nsga_matcher.match(targets[i])
        audio.save('./evaluation/nsga/nsga_prediction_%s.wav' % i)