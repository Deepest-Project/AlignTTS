
hparams = dict()

hparams.update({
    'language': 'Eng',
})

hparams.update({
    'metadata_path': '../Dataset/LJSpeech-1.1/metadata.csv',
    'wav_base_path': '../Dataset/LJSpeech-1.1/wavs',
})

hparams.update({
    'fs': 22500,
    'nsc_in_sec': 0.04,
    'hop_in_sec': 0.01,
    'n_mels': 80,  
})

hparams.update({
    'num_workers': 4,
    'batch_size': 64,
    # 'embedding_dim': 384,
    'embedding_dim': 768,
    # 'FFT_dim': 384,
    'FFT_dim': 768,
    'num_heads': 2,
    'kernel_size': 3,
    'num_FFT_blocks': 6,
    'num_MDN_layers': 6,
    'MDN_dim': 128,
    'MDN_dropout_rate': 0.2,
    'eps': 1e-8,
})

hparams.update({
    'nov_in_sec': hparams['nsc_in_sec'] - hparams['hop_in_sec'],
})

hparams.update({
    'nsc': int(hparams['nsc_in_sec'] * hparams['fs']),
    'hop': int(hparams['hop_in_sec'] * hparams['fs']),
    'nov': int(hparams['nov_in_sec'] * hparams['fs']),
})

print(hparams)