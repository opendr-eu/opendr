"""
This module implements computer vision methods.

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

# External Libraries
import numpy as np
import torch
from torchvision import transforms
import PIL
import cv2

# OpenDR Modules
from gui.fer import FER
from opendr.perception.facial_expression_recognition.image_based_facial_emotion_estimation.algorithm.utils import \
     image_processing, datasets
from opendr.perception.facial_expression_recognition.image_based_facial_emotion_estimation.algorithm.model.esr_9 \
    import ESR
from gui.grad_cam import GradCAM

# Haar cascade parameters
_HAAR_SCALE_FACTOR = 1.2
_HAAR_NEIGHBORS = 9
_HAAR_MIN_SIZE = (60, 60)

# Face detector method
_FACE_DETECTOR_HAAR_CASCADE = None

# Facial expression recognition network: Ensemble with Shared Representations (ESR)
_ESR_9 = None

# Saliency map generation: Grad-CAM
_GRAD_CAM = None


def detect_face(image):
    """
    Detects faces in an image.

    :param image: (ndarray) Raw input image.
    :return: (list) Tuples with coordinates of a detected face.
    """

    face_coordinates = []

    # Converts to greyscale
    greyscale_image = image_processing.convert_bgr_to_grey(image)

    face_coordinates = _haar_cascade_face_detection(greyscale_image, _HAAR_SCALE_FACTOR,
                                                    _HAAR_NEIGHBORS, _HAAR_MIN_SIZE)
    # Returns None if no face is detected
    return face_coordinates[0] if (len(face_coordinates) > 0 and (np.sum(face_coordinates[0]) > 0)) else None


def recognize_facial_expression(image, on_gpu, grad_cam):
    """
    Detects a face in the input image.
    If more than one face is detected, the biggest one is used.
    Afterwards, the detected face is fed to ESR-9 for facial expression recognition.
    The face detection phase relies on third-party methods and ESR-9 does not verify
    if a face is used as input or not (false-positive cases).

    :param on_gpu:
    :param image: (ndarray) input image.
    :return: An FER object with the components necessary for display.
    """

    to_return_fer = None
    saliency_maps = []

    # Detect face
    face_coordinates = detect_face(image)

    if face_coordinates is None:
        to_return_fer = FER(image)
    else:
        face = image[face_coordinates[0][1]:face_coordinates[1][1], face_coordinates[0][0]:face_coordinates[1][0], :]

        # Get device
        device = torch.device("cuda" if on_gpu else "cpu")

        # Pre_process detected face
        input_face = _pre_process_input_image(face)
        input_face = input_face.to(device)

        # Recognize facial expression
        # emotion_idx is needed to run Grad-CAM
        emotion, affect, emotion_idx = _predict(input_face, device)

        # Grad-CAM
        if grad_cam:
            saliency_maps = _generate_saliency_maps(input_face, emotion_idx, device)

        # Initialize GUI object
        to_return_fer = FER(image, face, face_coordinates, emotion, affect, saliency_maps)

    return to_return_fer


def _haar_cascade_face_detection(image, scale_factor, neighbors, min_size):
    """
    Face detection using the Haar Feature-based Cascade Classifiers (Viola and Jones, 2004).

    References:
    Viola, P. and Jones, M. J. (2004). Robust real-time face detection.
    International journal of computer vision, 57(2), 137-154.

    :param image: (ndarray) Raw image.
    :param scale_factor: Scale factor to resize input image.
    :param neighbors: Minimum number of bounding boxes to be classified as a face.
    :param min_size: Minimum size of the face bounding box.
    :return: (ndarray) Coordinates of the detected face.
    """
    global _FACE_DETECTOR_HAAR_CASCADE

    # Verifies if haar cascade classifiers are initialized
    if _FACE_DETECTOR_HAAR_CASCADE is None:
        _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("/face_detector/haar_cascade/frontal_face.xml")

    # Runs haar cascade classifiers
    faces = _FACE_DETECTOR_HAAR_CASCADE.detectMultiScale(image, scale_factor, neighbors, minSize=min_size)

    # Gets coordinates
    face_coordinates = [[[x, y], [x + w, y + h]] for (x, y, w, h) in faces] if not (faces is None) else []

    return np.array(face_coordinates)


def _pre_process_input_image(image):
    """
    Pre-processes an image for ESR-9.

    :param image: (ndarray)
    :return: (ndarray) image
    """

    image = image_processing.resize(image, ESR.INPUT_IMAGE_SIZE)
    image = PIL.Image.fromarray(image)
    image = transforms.Normalize(mean=ESR.INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=ESR.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)

    return image


def _predict(input_face, device):
    """
    Facial expression recognition. Classifies the pre-processed input image with ESR-9.

    :param input_face: (ndarray) input image.
    :param device: runs the classification on CPU or GPU
    :return: Lists of emotions and affect values including the ensemble predictions based on plurality.
    """

    global _ESR_9

    if _ESR_9 is None:
        _ESR_9 = ESR(device)

    to_return_emotion = []
    to_return_emotion_idx = []
    to_return_affect = None

    # Recognizes facial expression
    emotion, affect = _ESR_9(input_face)

    # Computes ensemble prediction for affect
    # Converts from Tensor to ndarray
    affect = np.array([a[0].cpu().detach().numpy() for a in affect])

    # Normalizes arousal
    affect[:, 1] = np.clip((affect[:, 1] + 1)/2.0, 0, 1)

    # Computes mean arousal and valence as the ensemble prediction
    ensemble_affect = np.expand_dims(np.mean(affect, 0), axis=0)

    # Concatenates the ensemble prediction to the list of affect predictions
    to_return_affect = np.concatenate((affect, ensemble_affect), axis=0)

    # Computes ensemble prediction concerning emotion labels
    # Converts from Tensor to ndarray
    emotion = np.array([e[0].cpu().detach().numpy() for e in emotion])

    # Gets number of classes
    num_classes = emotion.shape[1]

    # Computes votes and add label to the list of emotions
    emotion_votes = np.zeros(num_classes)
    for e in emotion:
        e_idx = np.argmax(e)
        to_return_emotion_idx.append(e_idx)
        to_return_emotion.append(datasets.AffectNetCategorical.get_class(e_idx))
        emotion_votes[e_idx] += 1

    # Concatenates the ensemble prediction to the list of emotion predictions
    to_return_emotion.append(datasets.AffectNetCategorical.get_class(np.argmax(emotion_votes)))

    return to_return_emotion, to_return_affect, to_return_emotion_idx


def _generate_saliency_maps(input_face, emotion_outputs, device):
    """
    Generates saliency maps for every branch in the ensemble with Grad-CAM.

    :param input_face: (ndarray) input image.
    :param device: runs the classification on CPU or GPU
    :return: (ndarray) Saliency maps.
    """

    global _GRAD_CAM, _ESR_9

    if _GRAD_CAM is None:
        _GRAD_CAM = GradCAM(_ESR_9, device)

    # Generate saliency map
    return _GRAD_CAM.grad_cam(input_face, emotion_outputs)
