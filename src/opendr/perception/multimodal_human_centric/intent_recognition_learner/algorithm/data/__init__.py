benchmarks = {
    'MIntRec': {
        'intent_labels': [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt',
            'Joke', 'Oppose',
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
                    'Prevent', 'Greet', 'Ask for help'
        ],
        'max_seq_length_text': 30,  # truth: 26
        'max_seq_length_video': 230,  # truth: 225
        'max_seq_length_audio': 480,  # truth: 477
        'video_feat_dim': 256,
        'audio_feat_dim': 768,
        'num_labels': 20
    }

}
