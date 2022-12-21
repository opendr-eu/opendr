"""
Demo script of the image-based facial emotion/expression estimation framework.

It has three main features:
Image: recognizes facial expressions in images.
Video: recognizes facial expressions in videos in a frame-based approach.
Webcam: connects to a webcam and recognizes facial expressions of the closest face detected
by a face detection algorithm.

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

# Standard Libraries
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from torchvision import transforms
import PIL
import cv2

# OpenDR Modules
from opendr.perception.facial_expression_recognition import FacialEmotionLearner, image_processing

INPUT_IMAGE_SIZE = (96, 96)
INPUT_IMAGE_NORMALIZATION_MEAN = [0.0, 0.0, 0.0]
INPUT_IMAGE_NORMALIZATION_STD = [1.0, 1.0, 1.0]


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


def detect_face(image):
    """
    Detects faces in an image.
    :param image: (ndarray) Raw input image.
    :return: (list) Tuples with coordinates of a detected face.
    """

    # Converts to greyscale
    greyscale_image = image_processing.convert_bgr_to_grey(image)

    # Runs haar cascade classifiers
    _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("./face_detector/frontal_face.xml")
    faces = _FACE_DETECTOR_HAAR_CASCADE.detectMultiScale(greyscale_image, scaleFactor=1.2, minNeighbors=9,
                                                         minSize=(60, 60))
    face_coordinates = [[[x, y], [x + w, y + h]] for (x, y, w, h) in faces] if not (faces is None) else []
    face_coordinates = np.array(face_coordinates)

    # Returns None if no face is detected
    return face_coordinates[0] if (len(face_coordinates) > 0 and (np.sum(face_coordinates[0]) > 0)) else None


def _pre_process_input_image(image):
    """
    Pre-processes an image for ESR-9.
    :param image: (ndarray)
    :return: (ndarray) image
    """

    image = image_processing.resize(image, INPUT_IMAGE_SIZE)
    image = PIL.Image.fromarray(image)
    image = transforms.Normalize(mean=INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)
    return image.numpy()


def _predict(learner, input_face):
    """
    Facial emotion/expression estimation. Classifies the pre-processed input image with FacialEmotionLearner.

    :param input_face: (ndarray) input image.
    :param device: runs the classification on CPU or GPU
    :param ensemble_size: number of branches in the network
    :return: Lists of emotions and affect values including the ensemble predictions based on plurality.
    """

    # Recognizes facial expression
    emotion, affect = learner.infer(input_face)
    # Converts from Tensor to ndarray
    affect = np.array([a.cpu().detach().numpy() for a in affect])
    to_return_affect = affect[0]  # a numpy array of valence and arousal values
    to_return_emotion = emotion[0]  # the emotion class with confidence tensor

    return to_return_emotion, to_return_affect


def recognize_facial_expression(learner, image, display):
    """
    Detects a face in the input image.
    If more than one face is detected, the biggest one is used.
    The detected face is fed to the _predict function which runs FacialEmotionLearner for facial emotion/expression
    estimation.
    :param image: (ndarray) input image.
    """

    # Detect face
    face_coordinates = detect_face(image)

    if face_coordinates is None:
        print("No face detected.")
    else:
        face = image[face_coordinates[0][1]:face_coordinates[1][1], face_coordinates[0][0]:face_coordinates[1][0], :]
        # Pre_process detected face
        input_face = _pre_process_input_image(face)
        # Recognize facial expression
        emotion, affect = _predict(learner, input_face=input_face)

        # display
        if display:
            image = cv2.putText(image, "Valence: %.2f" % affect[0], (10, 40 + 0 * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2, )
            image = cv2.putText(image, "Arousal: %.2f" % affect[1], (10, 40 + 1 * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2, )
            image = cv2.putText(image, emotion.description, (10, 40 + 2 * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2, )
        else:
            print('emotion:', emotion)
            print('valence, arousal:', affect)

    return image


def webcam(learner, camera_id, display, frames):
    """
    Receives images from a camera and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    if not image_processing.initialize_video_capture(camera_id):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether a webcam is working or not.")

    image_processing.set_fps(frames)

    try:
        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open():
            # Get a frame
            img, _ = image_processing.get_frame()
            img = None if (img is None) else recognize_facial_expression(learner, img, display)
            if display and img is not None:
                cv2.imshow('Result', img)
                cv2.waitKey(1)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    except KeyboardInterrupt:
        print("Keyboard interrupt event raised.")
    finally:
        image_processing.release_video_capture()
        if display:
            cv2.destroyAllWindows()


def image(learner, input_image_path, display):
    """
    Receives the full path to an image file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    img = image_processing.read(input_image_path)
    img = recognize_facial_expression(learner, img, display)
    if display:
        cv2.imshow('Result', img)
        cv2.waitKey(0)


def video(learner, input_video_path, display, frames):
    """
    Receives the full path to a video file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    if not image_processing.initialize_video_capture(input_video_path):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether working versions of ffmpeg or gstreamer is installed." +
                           "\nSupported file format: MPEG-4 (*.mp4).")
    image_processing.set_fps(frames)

    try:
        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open():
            # Get a frame
            img, timestamp = image_processing.get_frame()
            # Video has been processed
            if img is None:
                break
            else:  # Process frame
                img = None if (img is None) else recognize_facial_expression(learner, img, display)
                if display and img is not None:
                    cv2.imshow('Result', img)
                    cv2.waitKey(33)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        image_processing.release_video_capture()
        if display:
            cv2.destroyAllWindows()


def main():
    # Parser
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument("mode", help="select a method among 'image', 'video' or 'webcam' to run ESR-9.",
                        type=str, choices=["image", "video", "webcam"])
    parser.add_argument("-d", "--display", help="display the output of ESR-9.",
                        action="store_true")
    parser.add_argument("-i", "--input", help="define the full path to an image or video.",
                        type=str, default='')
    parser.add_argument("-es", "--ensemble_size",
                        help="define the size of the ensemble, the number of branches in the model",
                        type=int, default=9)
    parser.add_argument("--device", help="device to run on, either \'cpu\' or \'cuda\', defaults to \'cuda\'.",
                        default="cuda")
    parser.add_argument("-w", "--webcam_id",
                        help="define the webcam by 'id' to capture images in the webcam mode." +
                             "If none is selected, the default camera by the OS is used.",
                        type=int, default=-1)
    parser.add_argument("-f", "--frames", help="define frames of videos and webcam captures.",
                        type=int, default=5)

    args = parser.parse_args()

    learner = FacialEmotionLearner(device=args.device, ensemble_size=args.ensemble_size, dimensional_finetune=False,
                                   categorical_train=False)
    learner.init_model(num_branches=args.ensemble_size)
    model_path = learner.download(mode="pretrained")
    learner.load(args.ensemble_size, path_to_saved_network=model_path)

    # Calls to main methods
    if args.mode == "image":
        try:
            if is_none(args.input):
                args.input = learner.download(mode="demo_image")
            if is_none(args.input):
                raise RuntimeError("Error: 'input' is not valid. The argument 'input' is a mandatory "
                                   "field when image or video mode is chosen.")
            image(learner, args.input, args.display)
        except RuntimeError as e:
            print(e)
    elif args.mode == "video":
        try:
            if is_none(args.input):
                args.input = learner.download(mode="demo_video")
            if is_none(args.input):
                raise RuntimeError("Error: 'input' is not valid. The argument 'input' is a mandatory "
                                   "field when image or video mode is chosen.")
            video(learner, args.input, args.display, args.frames)
        except RuntimeError as e:
            print(e)
    elif args.mode == "webcam":
        try:
            webcam(learner, args.webcam_id, args.display, args.frames)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")
