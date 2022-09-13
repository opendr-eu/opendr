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
import time
import torch
import numpy as np
from torchvision import transforms
import PIL
import cv2

# OpenDR Modules
from opendr.perception.facial_expression_recognition import FacialEmotionLearner, ESR, image_processing, \
    file_maker, datasets
from gui.fer_display import FERDisplay
from gui.grad_cam import GradCAM
from gui import args_validation

# Haar cascade parameters
_HAAR_SCALE_FACTOR = 1.2
_HAAR_NEIGHBORS = 9
_HAAR_MIN_SIZE = (60, 60)


class FER:
    """
    This class implements the facial expression recognition object that contains the elements
    to be displayed on the screen such as an input image and ESR-9's outputs.
    """

    def __init__(self, image=None, face_image=None, face_coordinates=None,
                 list_emotion=None, list_affect=None, list_grad_cam=None):
        """
        Initialize FER object.
        """
        self.input_image = image
        self.face_coordinates = face_coordinates
        self.face_image = face_image
        self.list_emotion = list_emotion
        self.list_affect = list_affect
        self._list_grad_cam = list_grad_cam

    def get_grad_cam(self, i):
        if (self._list_grad_cam is None) or (len(self._list_grad_cam) == 0):
            return None
        else:
            return self._list_grad_cam[i]


def detect_face(image):
    """
    Detects faces in an image.
    :param image: (ndarray) Raw input image.
    :return: (list) Tuples with coordinates of a detected face.
    """

    # Converts to greyscale
    greyscale_image = image_processing.convert_bgr_to_grey(image)

    # Runs haar cascade classifiers
    _FACE_DETECTOR_HAAR_CASCADE = cv2.CascadeClassifier("/face_detector/frontal_face.xml")
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

    image = image_processing.resize(image, ESR.INPUT_IMAGE_SIZE)
    image = PIL.Image.fromarray(image)
    image = transforms.Normalize(mean=ESR.INPUT_IMAGE_NORMALIZATION_MEAN,
                                 std=ESR.INPUT_IMAGE_NORMALIZATION_STD)(transforms.ToTensor()(image)).unsqueeze(0)
    return image


def _predict(input_face, device, model_path, ensemble_size):
    """
    Facial emotion/expression estimation. Classifies the pre-processed input image with FacialEmotionLearner.

    :param input_face: (ndarray) input image.
    :param device: runs the classification on CPU or GPU
    :param model_path: path to the saved network
    :param model_path: number of branches in the network
    :return: Lists of emotions and affect values including the ensemble predictions based on plurality.
    """

    learner = FacialEmotionLearner(device=device, ensemble_size=ensemble_size)
    learner.init_model(num_branches=ensemble_size)
    learner.load(ensemble_size, path_to_saved_network=model_path)

    to_return_emotion = []
    to_return_emotion_idx = []

    # Recognizes facial expression
    emotion, affect = learner.infer(input_face)

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

    return to_return_emotion, to_return_affect, to_return_emotion_idx, learner.model


def recognize_facial_expression(image, on_gpu, grad_cam, model_path, ensemble_size):
    """
    Detects a face in the input image.
    If more than one face is detected, the biggest one is used.
    The detected face is fed to the _predict function which runs FacialEmotionLearner for facial emotion/expression
    estimation.

    :param image: (ndarray) input image.
    :return: A FER object with the components necessary for display.
    """

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
        emotion, affect, emotion_idx, model = _predict(input_face, device, model_path, ensemble_size)

        # Grad-CAM
        if grad_cam:
            _GRAD_CAM = GradCAM(model, device)
            saliency_maps = _GRAD_CAM.grad_cam(input_face, emotion_idx)

        # Initialize GUI object
        to_return_fer = FER(image, face, face_coordinates, emotion, affect, saliency_maps)

    return to_return_fer


def webcam(camera_id, display, gradcam, output_csv_file, screen_size, device, frames, no_plot, pretrained_model_path,
           ensemble_size):
    """
    Receives images from a camera and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_display = None
    write_to_file = not (output_csv_file is None)
    starting_time = time.time()

    if not image_processing.initialize_video_capture(camera_id):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether a webcam is working or not." +
                           "In linux, you can use Cheese for testing.")

    image_processing.set_fps(frames)

    # Initialize screen
    if display:
        fer_display = FERDisplay(screen_size=screen_size, display_graph_ensemble=(not no_plot))
    else:
        print("Press 'Ctrl + C' to quit.")

    try:
        if write_to_file:
            file_maker.create_file(output_csv_file, str(time.time()))

        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open() and ((not display) or (display and fer_display.is_running())):
            # Get a frame
            img, _ = image_processing.get_frame()
            fer = None if (img is None) else recognize_facial_expression(img, device, gradcam,
                                                                         pretrained_model_path, ensemble_size)

            # Display blank screen if no face is detected, otherwise,
            # display detected faces and perceived facial expression labels
            if display:
                fer_display.update(fer)
                fer_display.show()

            if write_to_file:
                file_maker.write_to_file(fer, time.time() - starting_time)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    except KeyboardInterrupt:
        print("Keyboard interrupt event raised.")
    finally:
        image_processing.release_video_capture()
        if display:
            fer_display.quit()
        if write_to_file:
            file_maker.close_file()


def image(input_image_path, display, gradcam, output_csv_file, screen_size, device, pretrained_model_path,
          ensemble_size):
    """
    Receives the full path to an image file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    write_to_file = not (output_csv_file is None)
    img = image_processing.read(input_image_path)

    # Call FER method
    fer = recognize_facial_expression(img, device, gradcam, pretrained_model_path, ensemble_size)

    if write_to_file:
        file_maker.create_file(output_csv_file, input_image_path)
        file_maker.write_to_file(fer, 0.0)
        file_maker.close_file()

    if display:
        fer_display = FERDisplay(screen_size=screen_size,
                                 display_graph_ensemble=False)
        fer_display.update(fer)
        while fer_display.is_running():
            fer_display.show()
        fer_display.quit()


def video(input_video_path, display, gradcam, output_csv_file, screen_size,
          device, frames, no_plot, pretrained_model_path, ensemble_size):
    """
    Receives the full path to a video file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_display = None
    write_to_file = not (output_csv_file is None)

    if not image_processing.initialize_video_capture(input_video_path):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether working versions of ffmpeg or gstreamer is installed." +
                           "\nSupported file format: MPEG-4 (*.mp4).")

    image_processing.set_fps(frames)

    # Initialize screen
    if display:
        fer_display = FERDisplay(screen_size=screen_size, display_graph_ensemble=(not no_plot))

    try:
        if write_to_file:
            file_maker.create_file(output_csv_file, input_video_path)

        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open() and ((not display) or (display and fer_display.is_running())):
            # Get a frame
            img, timestamp = image_processing.get_frame()

            # Video has been processed
            if img is None:
                break
            else:  # Process frame
                fer = None if (img is None) else recognize_facial_expression(img, device, gradcam,
                                                                             pretrained_model_path,
                                                                             ensemble_size)
                # Display blank screen if no face is detected, otherwise,
                # display detected faces and perceived facial expression labels
                if display:
                    fer_display.update(fer)
                    fer_display.show()

                if write_to_file:
                    file_maker.write_to_file(fer, timestamp)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        image_processing.release_video_capture()
        if display:
            fer_display.quit()
        if write_to_file:
            file_maker.close_file()


def main():
    # Parser
    parser = argparse.ArgumentParser(description='test', formatter_class=RawTextHelpFormatter)
    parser.add_argument("mode", help="select a method among 'image', 'video' or 'webcam' to run ESR-9.",
                        type=str, choices=["image", "video", "webcam"])
    parser.add_argument("-d", "--display", help="display the output of ESR-9.",
                        action="store_true")
    parser.add_argument("-g", "--gradcam", help="run grad-CAM and displays the salience maps.",
                        action="store_true")
    parser.add_argument("-i", "--input", help="define the full path to an image or video.",
                        type=str)
    parser.add_argument("-o", "--output",
                        help="create and write ESR-9's outputs to a CSV file. The file is saved in a folder defined "
                             "by this argument (ex. '-o ./' saves the file with the same name as the input file "
                             "in the working directory).",
                        type=str)
    parser.add_argument("-pre", "--pretrained", help="define the full path to the pretrained model weights.",
                        type=str)
    parser.add_argument("-es", "--ensemble_size",
                        help="define the size of the ensemble, the number of branches in the model",
                        type=int, default=9)
    parser.add_argument("-s", "--size",
                        help="define the size of the window: \n1 - 1920 x 1080;\n2 - 1440 x 900;\n3 - 1024 x 768.",
                        type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("-c", "--cuda", help="run on GPU.",
                        action="store_true")
    parser.add_argument("-w", "--webcam_id",
                        help="define the webcam by 'id' to capture images in the webcam mode." +
                             "If none is selected, the default camera by the OS is used.",
                        type=int, default=-1)
    parser.add_argument("-f", "--frames", help="define frames of videos and webcam captures.",
                        type=int, default=5)
    parser.add_argument("-np", "--no_plot", help="do not display activation and (un)pleasant graph",
                        action="store_true", default=False)

    args = parser.parse_args()

    # Calls to main methods
    if args.mode == "image":
        try:
            args_validation.validate_image_video_mode_arguments(args)
            image(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.pretrained, args.ensemble_size)
        except RuntimeError as e:
            print(e)
    elif args.mode == "video":
        try:
            args_validation.validate_image_video_mode_arguments(args)
            video(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.frames, args.no_plot, args.pretrained, args.ensemble_size)
        except RuntimeError as e:
            print(e)
    elif args.mode == "webcam":
        try:
            args_validation.validate_webcam_mode_arguments(args)
            webcam(args.webcam_id, args.display, args.gradcam, args.output,
                   args.size, args.cuda, args.frames, args.no_plot, args.pretrained, args.ensemble_size)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")
