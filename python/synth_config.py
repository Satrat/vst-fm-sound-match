import spiegelib as spgl

def configWTSynth(synth_path, synth_state):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=1.0, render_length_secs=1.0)
    synth.save_state("./synth_params_camomile/wt_synth_init.json")

    # turn off LFO and misc unneeded params
    overridden_parameters = [
        (5, 0.478), # bpm Fix to 120
        (6, 0.0), # clipDist none
        (27, 0.0), # pitchTrack none
        (22, 0.0), #levelPinkNoise no noise
        (25, 0.5), #masterVol midpoint
        (2, 0.0), # ampMod no amplitude modulation
        (10, 0.5), # filterAmmLFO1 filter at 0Hz
        (16, 0.0), # freqLFO1 none 
        (17, 0.0), # freqLFO2 none
        (30, 0.0), # wave1ModAmm none
        (31, 0.0), # wave2ModAmm none
        (32, 0.0), # wave3ModAmm none
        (33, 0.0), # waveLFO1 default wave
        (34, 0.0) # waveLFO2 default wave
    ]

    # Set overridden parameters in synth
    synth.set_overridden_parameters(overridden_parameters)
    synth.save_state(synth_state)


def configDexed(synth_path, synth_state="./synth_params/dexed_simple_fm.json"):
    synth = spgl.synth.SynthVST(synth_path, note_length_secs=1.0, render_length_secs=1.0)
    synth.save_state("./synth_params/dexed_simple_fm_init.json")

    # The algorithm sets the arrangement of the FM operators. 
    # There are 32 differen arrangements available in Dexed.
    # This selects the first algorithm. This algorithm sets
    # the second operator to frequency modulate the first.
    algorithm_number = 1
    alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001

    overridden_parameters = [
        (0, 1.0), # Filter Cutoff (Fully open)
        (1, 0.0), # Filter Resonance
        (2, 1.0), # Output Gain 
        (3, 0.5), # Master Tuning (Center is 0)
        (4, alg), # Operator configuration
        (5, 0.0), # Feedback
        (6, 1.0), # Key Sync Oscillators 
        (7, 0.0), # LFO Speed
        (8, 0.0), # LFO Delay
        (9, 0.0), # LFO Pitch Modulation Depth
        (10, 0.0),# LFO Amplitude Modulation Depth
        (11, 0.0),# LFO Key Sync
        (12, 0.0),# LFO Waveform
        (13, 0.5),# Middle C Tuning
    ]

    # Turn off all pitch modulation parameters
    overridden_parameters.extend([(i, 0.0) for i in range(14, 23)])

    # Turn Operator 1 into a simple sine wave with no envelope
    overridden_parameters.extend([
        (23, 0.9), # Operator 1 Attack Rate
        (24, 0.9), # Operator 1 Decay Rate
        (25, 0.9), # Operator 1 Sustain Rate
        (26, 0.9), # Operator 1 Release Rate
        (27, 1.0), # Operator 1 Attack Level
        (28, 1.0), # Operator 1 Decay Level
        (29, 1.0), # Operator 1 Sustain Level
        (30, 0.0), # Operator 1 Release Level
        (31, 1.0), # Operator 1 Gain
        (32, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (33, 0.5), # Operator 1 Coarse Tuning
        (34, 0.0), # Operator 1 Fine Tuning
        (35, 0.5), # Operator 1 Detune
        (36, 0.0), # Operator 1 Env Scaling Param
        (37, 0.0), # Operator 1 Env Scaling Param
        (38, 0.0), # Operator 1 Env Scaling Param
        (39, 0.0), # Operator 1 Env Scaling Param
        (40, 0.0), # Operator 1 Env Scaling Param
        (41, 0.0), # Operator 1 Env Scaling Param
        (42, 0.0), # Operator 1 Mod Sensitivity
        (43, 0.0), # Operator 1 Key Velocity
        (44, 1.0), # Operator 1 On/Off switch
    ])

    # Override some of Operator 2 parameters
    overridden_parameters.extend([
        (45, 0.9), # Operator 2 Attack Rate (No attack on operator 2)
        (49, 1.0), # Operator 2 Attack Level
        (53, 1.0), # Operator 2 Gain (Operator 2 always outputs)
        (54, 0.0), # Operator 1 Mode (1.0 is Fixed Frequency)
        (58, 0.0), # Operator 1 Env Scaling Param
        (59, 0.0), # Operator 1 Env Scaling Param
        (60, 0.0), # Operator 1 Env Scaling Param
        (61, 0.0), # Operator 1 Env Scaling Param
        (62, 0.0), # Operator 1 Env Scaling Param
        (63, 0.0), # Operator 1 Env Scaling Param
        (64, 0.0), # Operator 1 Mod Sensitivity
        (65, 0.0), # Operator 1 Key Velocity
        (66, 1.0), # Operator 1 On/Off switch
    ])

    # Override operators 3 through 6
    overridden_parameters.extend([(i, 0.0) for i in range(67, 155)])

    # Set overridden parameters in synth
    synth.set_overridden_parameters(overridden_parameters)
    synth.save_state(synth_state)