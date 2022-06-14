"""
This module validates users' inputs.

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""


def is_none(x):
    """
    Verifies is the string 'x' is none.
    :param x: (string)
    :return: (bool)
    """
    if (x is None) or ((type(x) == str) and (x.strip() == "")):
        return True
    else:
        return False


def validate_image_video_mode_arguments(args):
    """
    User input validation.
    Validates the arguments passed by users through the terminal when 'mode' is set to
    'image' or 'video' in 'main_esr9.py'.

    :param args: arguments passed through the terminal.
    :return: void
    """

    # The argument input is mandatory
    if is_none(args.input):
        raise RuntimeError("Error: 'input' is not valid. The argument 'input' is a mandatory "
                           "field when image or video mode is chosen.")

    # If 'gradcam' is True and 'display' is mandatory
    if args.gradcam and (not args.display):
        raise RuntimeError("Error: 'gradcam' requires 'display' equals True.")

    # If 'display' is False, 'output' is mandatory
    if (not args.display) and is_none(args.output):
        raise RuntimeError("Error: 'output' is not valid. The argument 'output' is a mandatory "
                           "field when display is False.")


def validate_webcam_mode_arguments(args):
    """
    Validates the arguments passed by users through the terminal when 'mode' is set to 'webcam' in 'main_esr9.py'.

    :param args: arguments passed through the terminal.
    :return: void
    """
    # If 'gradcam' is True and 'display' is mandatory
    if args.gradcam and (not args.display):
        raise RuntimeError("Error: 'gradcam' requires 'display' equals True.")

    # If 'display' is False, 'output' is mandatory
    if (not args.display) and is_none(args.output):
        raise RuntimeError("Error: 'output' is not valid. The argument 'output' is a mandatory "
                           "field when display is False.")
