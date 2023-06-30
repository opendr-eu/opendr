from opendr.perception.multimodal_human_centric import IntentRecognitionLearner
import argparse

LABELS = [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort',
            'Care', 'Inform', 'Advise', 'Arrange', 'Introduce',
            'Leave', 'Prevent', 'Greet', 'Ask for help'
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--text_backbone",
                        help="Text backbone: ['bert-base-uncased' | 'albert-base-v2' | \
                            'bert-small' | 'bert-mini' | 'bert-tiny']",
                        type=str, default="bert-base-uncased")
    parser.add_argument("--cache_path", help="Cache path for tokenizer files", type=str, default="cache")
    args = parser.parse_args()

    if args.text_backbone == 'bert-small':
        args.text_backbone = 'prajjwal1/bert-small'
    elif args.text_backbone == 'bert-mini':
        args.text_backbone = 'prajjwal1/bert-mini'
    elif args.text_backbone == 'bert-tiny':
        args.text_backbone = 'prajjwal1/bert-tiny'

    modality = 'language'

    learner = IntentRecognitionLearner(text_backbone=args.text_backbone, mode=modality,
                                       device=args.device, log_path='logs',
                                       cache_path=args.cache_path, results_path='result',
                                       output_path='outputs')
    # learner.download('pretrained_models/')
    # learner.load('pretrained_models/', args.text_backbone + '.pth')

    while True:
        raw_text = input('Enter text: ')
        pred = learner.infer({'text': raw_text}, modality='language')
        for utt in pred:
            print(LABELS[utt.data], utt.confidence)
