import importlib
import torch


def find_trainer_using_name(trainer_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    trainer_filename = "trainers." + trainer_name + "_trainer"
    modellib = importlib.import_module(trainer_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.
    trainer = None
    target_model_name = trainer_name.replace('_', '') + 'trainer'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            trainer = cls

    if trainer is None:
        print("In %s.py, there should be a trainer name that matches %s in lowercase." % (trainer_filename, target_model_name))
        exit(0)

    return trainer


def get_option_setter(trainer_name):
    model_class = find_trainer_using_name(trainer_name)
    return model_class.modify_commandline_options


def create_trainer(opt):
    trainer = find_trainer_using_name(opt.trainer)
    instance = trainer(opt)
    print("trainer was created")
    return instance