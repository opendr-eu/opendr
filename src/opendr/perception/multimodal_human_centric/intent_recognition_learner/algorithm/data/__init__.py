'''
This code is based on https://github.com/thuiar/MIntRec under MIT license:

MIT License

Copyright (c) 2022 Hanlei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''
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
