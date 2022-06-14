
"""
This module creates, writes and closes files.
Modified based on:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

import os

# Default values
_NUM_BRANCHES = 9
_HEADER = "Timestamp(sec),Ensemble_Emotion,Ensemble_Arousal,Ensemble_Valence"
for i in range(_NUM_BRANCHES):
    _HEADER += ",Branch_{}_Emotion,Branch_{}_Arousal,Branch_{}_Valence".format(i, i, i)
_HEADER += "\n"

# File object
_FILE = None


def create_file(directory, file_name):
    global _FILE

    if not (_FILE is None) and not _FILE.closed:
        raise RuntimeError("A file is open.")

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)

        name = file_name.split(os.sep)[-1]
        name = name.split(".")[0] + ".csv"

        full_path = os.path.join(directory, name)

        _FILE = open(full_path, "w")
        _FILE.write(_HEADER)

        print("Output file saved at: {}".format(full_path))
    except RuntimeError as e:
        _FILE = None
        print("Error on trying to create a csv file.")
        raise e


def write_to_file(fer, timestamp):
    global _FILE

    try:
        line = "{}".format(timestamp)

        if (fer is None) or (fer.list_emotion is None) or (fer.list_affect is None):
            for b in range(_NUM_BRANCHES + 1):
                line += ",None,None,None"
        else:
            line += ",{},{},{}".format(fer.list_emotion[-1], fer.list_affect[-1][1], fer.list_affect[-1][0])
            for b in range(_NUM_BRANCHES):
                line += ",{},{},{}".format(fer.list_emotion[b], fer.list_affect[b][1], fer.list_affect[b][0])

        line += "\n"
        _FILE.write(line)
    except RuntimeError as e:
        print("Error on trying to write to file.")
        raise e


def close_file():
    global _FILE

    if not (_FILE is None):
        _FILE.close()

    _FILE = None
