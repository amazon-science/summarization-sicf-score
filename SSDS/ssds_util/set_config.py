

def expand_config(config, data_args): # generation config
    # for DialogLED
    config.num_beams = 6
    if 'proSAMSUM' in data_args.train_file or 'newSAMSUM' in data_args.train_file: 
        config.max_length = 96
        config.min_length = 4
    elif 'DIALOGSUM' in data_args.train_file:
        config.max_length = 96
        config.min_length = 4
    elif 'newTODSUM' in data_args.train_file:
        config.max_length = 140
        config.min_length = 15
    else:
        raise ValueError('The data_args.train_file has unexpected preset parameter')


    config.length_penalty = 2.0 # commented out by myself
    config.no_repeat_ngram_size = 3
    return config