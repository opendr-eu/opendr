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


import os
import torch
import numpy as np
import pandas as pd
import random
from .metrics import Metrics


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_output_path(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pred_output_path = os.path.join(args.output_path, args.logger_name)
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)

    model_path = os.path.join(pred_output_path, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return pred_output_path, model_path


def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)


def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file


def save_model(model, model_dir, accs={}, name='pytorch_model.pth'):
    save_model = model.module if hasattr(model, 'module') else model
    model_file = os.path.join(model_dir, name)
    save_dict = {'state_dict': save_model.state_dict()}
    save_dict.update(accs)
    torch.save(save_dict, model_file)


def restore_model(model, model_dir, name='pytorch_model.pth'):
    output_model_file = os.path.join(model_dir, name)
    checkpoint = torch.load(output_model_file)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def save_results(args, test_results, debug_args=None, suff='', results_file_name='results.csv'):

    pred_labels_path = os.path.join(args.pred_output_path, 'y_pred_'+suff+'.npy')
    np.save(pred_labels_path, test_results['y_pred'])

    true_labels_path = os.path.join(args.pred_output_path, 'y_true_'+suff+'.npy')
    np.save(true_labels_path, test_results['y_true'])

    if 'features' in test_results.keys():
        features_path = os.path.join(args.pred_output_path, 'features_'+suff+'.npy')
        np.save(features_path, test_results['features'])

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    results = {}
    metrics = Metrics(args)
    for key in metrics.eval_metrics:
        results[key] = round(test_results[key] * 100, 2)

    _vars = [args.text_backbone, args.seed, args.logger_name]
    _names = ['dataset',  'method', 'text_backbone', 'seed', 'logger_name']

    if debug_args is not None:
        _vars.extend([args[key] for key in debug_args.keys()])
        _names.extend(debug_args.keys())

    vars_dict = {k: v for k, v in zip(_names, _vars)}
    results = dict(results, **vars_dict)

    keys = list(results.keys())
    values = list(results.values())

    results_path = os.path.join(args.results_path, results_file_name)

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = pd.concat([df1, pd.DataFrame(new)], ignore_index=True)
        df1.to_csv(results_path, index=False)
