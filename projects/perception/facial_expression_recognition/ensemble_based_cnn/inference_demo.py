"""
Main script of the facial expression recognition framework.

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

# OpenDR Modules
from controller import cvalidation, cvision
from opendr.perception.facial_expression_recognition.ensemble_based_cnn.algorithm.utils import \
    file_maker, image_processing
from gui.fer_demo import FERDemo


def webcam(camera_id, display, gradcam, output_csv_file, screen_size, device, frames, no_plot, face_detection):
    """
    Receives images from a camera and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_demo = None
    write_to_file = not (output_csv_file is None)
    starting_time = time.time()

    if not image_processing.initialize_video_capture(camera_id):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether a webcam is working or not." +
                           "In linux, you can use Cheese for testing.")

    image_processing.set_fps(frames)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_graph_ensemble=(not no_plot))
    else:
        print("Press 'Ctrl + C' to quit.")

    try:
        if write_to_file:
            file_maker.create_file(output_csv_file, str(time.time()))

        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            img, _ = image_processing.get_frame()

            fer = None if (img is None) else cvision.recognize_facial_expression(img, device, face_detection, gradcam)

            # Display blank screen if no face is detected, otherwise,
            # display detected faces and perceived facial expression labels
            if display:
                fer_demo.update(fer)
                fer_demo.show()

            if write_to_file:
                file_maker.write_to_file(fer, time.time() - starting_time)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    except KeyboardInterrupt as qe:
        print("Keyboard interrupt event raised.")
    finally:
        image_processing.release_video_capture()

        if display:
            fer_demo.quit()

        if write_to_file:
            file_maker.close_file()


def image(input_image_path, display, gradcam, output_csv_file, screen_size, device, face_detection):
    """
    Receives the full path to an image file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    write_to_file = not (output_csv_file is None)
    img = image_processing.read(input_image_path)

    # Call FER method
    fer = cvision.recognize_facial_expression(img, device, face_detection, gradcam)

    if write_to_file:
        file_maker.create_file(output_csv_file, input_image_path)
        file_maker.write_to_file(fer, 0.0)
        file_maker.close_file()

    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_graph_ensemble=False)
        fer_demo.update(fer)
        while fer_demo.is_running():
            fer_demo.show()
        fer_demo.quit()


def video(input_video_path, display, gradcam, output_csv_file, screen_size,
          device, frames, no_plot, face_detection):
    """
    Receives the full path to a video file and recognizes
    facial expressions of the closets face in a frame-based approach.
    """

    fer_demo = None
    write_to_file = not (output_csv_file is None)

    if not image_processing.initialize_video_capture(input_video_path):
        raise RuntimeError("Error on initializing video capture." +
                           "\nCheck whether working versions of ffmpeg or gstreamer is installed." +
                           "\nSupported file format: MPEG-4 (*.mp4).")

    image_processing.set_fps(frames)

    # Initialize screen
    if display:
        fer_demo = FERDemo(screen_size=screen_size,
                           display_graph_ensemble=(not no_plot))

    try:
        if write_to_file:
            file_maker.create_file(output_csv_file, input_video_path)

        # Loop to process each frame from a VideoCapture object.
        while image_processing.is_video_capture_open() and ((not display) or (display and fer_demo.is_running())):
            # Get a frame
            img, timestamp = image_processing.get_frame()

            # Video has been processed
            if img is None:
                break
            else:  # Process frame
                fer = None if (img is None) else cvision.recognize_facial_expression(img,
                                                                                     device,
                                                                                     face_detection,
                                                                                     gradcam)
                # Display blank screen if no face is detected, otherwise,
                # display detected faces and perceived facial expression labels
                if display:
                    fer_demo.update(fer)
                    fer_demo.show()

                if write_to_file:
                    file_maker.write_to_file(fer, timestamp)

    except Exception as e:
        print("Error raised during video mode.")
        raise e
    finally:
        image_processing.release_video_capture()

        if display:
            fer_demo.quit()

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
            cvalidation.validate_image_video_mode_arguments(args)
            image(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.face_detection)
        except RuntimeError as e:
            print(e)
    elif args.mode == "video":
        try:
            cvalidation.validate_image_video_mode_arguments(args)
            video(args.input, args.display, args.gradcam, args.output,
                  args.size, args.cuda, args.frames, args.no_plot, args.face_detection)
        except RuntimeError as e:
            print(e)
    elif args.mode == "webcam":
        try:
            cvalidation.validate_webcam_mode_arguments(args)
            webcam(args.webcam_id, args.display, args.gradcam, args.output,
                   args.size, args.cuda, args.frames, args.no_plot, args.face_detection)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    print("Processing...")
    main()
    print("Process has finished!")


