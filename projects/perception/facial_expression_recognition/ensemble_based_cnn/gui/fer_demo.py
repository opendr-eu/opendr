"""
GUI of the facial expression recognition (FER) demo

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

# External Libraries
import numpy as np
import cv2

# OpenDR Modules
from opendr.perception.facial_expression_recognition.ensemble_based_cnn.algorithm.utils import image_processing


class FERDemo:
    """
    This class implements the GUI of the facial expression recognition (FER) demo.
    """

    # Default values
    _DEFAULT_SCREEN_SIZE_ID = 1
    _DEFAULT_WINDOW_NAME = "Siqueira_et_al_AAAI_2020"
    _DEFAULT_DISPLAY_GRAPH_ENSEMBLE = True

    # Display
    _SCREEN_SIZE = [(1920, 1080), (1440, 900), (1024, 768)]
    _TEXT_PARAM_SCALE = [0.9, 0.8, 0.7]
    _TEXT_PARAM_THICKNESS = [2, 2, 2]
    _INPUT_IMAGE_SCALE_MAX = 0.9
    _INPUT_IMAGE_SCALE_MIN = 0.4

    # Display: blocks
    _BLOCK_NUM_BLOCKS = 10  # Ensemble size
    _BLOCK_INIT_POS_TEXT_NETWORK = [(0, 30), (0, 20), (0, 20)]

    _BLOCK_INIT_POS_IMAGE = [(4, 170), (4, 145), (4, 125)]
    _BLOCK_IMAGE_SIZE = [(100, 100), (75, 75), (60, 60)]

    _BLOCK_INIT_POS_TEXT_EMOTION = [(300, 55), (240, 45), (195, 40)]
    _BLOCK_INIT_POS_TEXT_AROUSAL = [(470, 40), (380, 25), (300, 25)]
    _BLOCK_INIT_POS_TEXT_VALENCE = [(470, 85), (380, 65), (300, 55)]

    _BLOCK_INIT_POS_BAR_AROUSAL = [(550, 15), (450, 5), (350, 7)]
    _BLOCK_FINAL_POS_BAR_AROUSAL = [(920, 45), (700, 30), (500, 27)]

    _BLOCK_INIT_POS_BAR_VALENCE = [(550, 60), (450, 45), (350, 42)]
    _BLOCK_FINAL_POS_BAR_VALENCE = [(920, 90), (700, 70), (500, 62)]

    # Ensemble
    _BLOCK_INIT_POS_TEXT_NETWORK_ENSEMBLE = [(10, 50), (10, 40), (10, 40)]

    _BLOCK_INIT_POS_IMAGE_ENSEMBLE = [(80, 10), (60, 10), (60, 10)]
    _BLOCK_IMAGE_SIZE_ENSEMBLE = [(200, 200), (150, 150), (120, 120)]

    _BLOCK_INIT_POS_TEXT_EMOTION_ENSEMBLE = [(10, 350), (10, 270), (10, 220)]
    _BLOCK_INIT_POS_TEXT_ACTIVATION = [(10, 420), (10, 330), (10, 260)]
    _BLOCK_INIT_POS_TEXT_PLEASANT = [(10, 500), (10, 410), (10, 320)]
    _BLOCK_INIT_POS_TEXT_UNPLEASANT = [(10, 580), (10, 490), (10, 380)]

    _BLOCK_INIT_POS_BAR_ACTIVATION = [(10, 435), (10, 345), (10, 270)]
    _BLOCK_FINAL_POS_BAR_ACTIVATION = [(600, 465), (450, 370), (300, 290)]

    _BLOCK_INIT_POS_BAR_PLEASANT = [(10, 515), (10, 425), (10, 330)]
    _BLOCK_FINAL_POS_BAR_PLEASANT = [(600, 545), (450, 450), (300, 350)]

    _BLOCK_INIT_POS_BAR_UNPLEASANT = [(10, 595), (10, 505), (10, 390)]
    _BLOCK_FINAL_POS_BAR_UNPLEASANT = [(600, 635), (450, 530), (300, 410)]

    _BLOCK_INIT_POS_GRAPH = [(660, 10), (580, 10), (460, 10)]
    _BLOCK_SAMPLE_GRAPH = 16
    _BLOCK_THICKNESS_GRAPH = [3, 3, 3]
    _BLOCK_FONT_SIZE_GRAPH = [14, 12, 10]
    _BLOCK_OFFSET_GRAPH = [60, 60, 40]
    _BLOCK_SIZE_GRAPH = [(8, 3.2), (7, 3), (5, 2.5)]

    # Display: maximum values
    _MAX_AROUSAL = 1.0
    _MAX_VALENCE = 1.0

    # Colours GREYSCALE
    _COLOUR_G_DARK_GREY = 50

    # Colours BGR
    _COLOUR_BGR_GREEN = (0, 255, 0)
    _COLOUR_BGR_RED = (0, 0, 255)
    _COLOUR_BGR_WHITE = (255, 255, 255)
    _COLOUR_BGR_BLACK = (0, 0, 0)
    _COLOUR_BGR_ORANGE = (0, 125, 255)
    _COLOUR_BGR_BLUE = (255, 0, 0)
    _COLOUR_BGR_DARK_RED = (0, 0, 130)
    _COLOUR_BGR_DARK_GREEN = (60, 130, 0)
    _COLOUR_BGR_DARK_BLUE = (130, 60, 0)
    _COLOUR_BGR_DARK_GREY = (50, 50, 50)

    # Messages
    _TEXT_BLANK_INPUT = "No frame to process."
    _TEXT_NO_FACE = "No face has been detected."
    _TEXT_ENSEMBLE = "Ensemble:"
    _TEXT_BRANCH = "Branch {}:"
    _TEXT_AROUSAL = "Aro:"
    _TEXT_VALENCE = "Val:"
    _TEXT_ACTIVATION = "Activation:"
    _TEXT_PLEASANT = "Pleasant:"
    _TEXT_UNPLEASANT = "Unpleasant:"
    _TEXT_ACTIVATION_WITHOUT_TWO_DOTS = "Activation"
    _TEXT_PLEASANT_UNPLEASANT = "Pleasant / Unpleasant"

    def __init__(self, window_name=_DEFAULT_WINDOW_NAME, screen_size=_DEFAULT_SCREEN_SIZE_ID,
                 display_graph_ensemble=_DEFAULT_DISPLAY_GRAPH_ENSEMBLE):
        """
        Initialize GUI of the FER demo.
        :param window_name: (string) The name of the window
        :param screen_size: ((int, int)) Tuple of int values for width and height, respectively.
        """

        # Screen components
        self._fer = None
        self._input_image = None
        self._background = None
        self._plot_arousal = []
        self._plot_valence = []

        # Screen
        self._window_name = window_name
        self._screen_size = screen_size - 1
        self._width, self._height = FERDemo._SCREEN_SIZE[self._screen_size]
        self._display_graph_ensemble = display_graph_ensemble

        # Container parameters
        self._container_width, self._container_height = (int(self._width // 2), int(self._height))
        self._container_center_position = np.array([self._container_width // 2, self._container_height // 2],
                                                   dtype=np.int)
        self._input_container = None
        self._output_container = None
        self._input_container_initial_position = np.array([0, 0], dtype=np.int)
        self._output_container_initial_position = np.array([0, self._width // 2], dtype=np.int)

        # Output blocks
        self._output_block_height = (self._container_height // FERDemo._BLOCK_NUM_BLOCKS)
        self._output_block_height_ensemble = self._container_height
        self._output_block_width = self._container_width

        # Screen initialization
        self._draw_background()
        self._screen = self._get_container(0, 0, self._height, self._width)
        self._blank_screen()

        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

    def _blank_screen(self):
        """
        Create a blank screen without an input image and outputs.
        """
        self._draw_input_container(True)
        self._draw_output_container(True)
        self._draw_screen()

    def _draw_screen(self):
        self._screen[:, :self._output_container_initial_position[1], :] = self._input_container
        self._screen[:, self._output_container_initial_position[1]:, :] = self._output_container

    def _draw_input_container(self, is_blank):
        self._input_container = self._get_container(0, 0, self._container_height, self._container_width)

        if is_blank:
            image_processing.draw_text(self._input_container,
                             FERDemo._TEXT_BLANK_INPUT,
                             self._container_center_position - 60,
                             FERDemo._COLOUR_BGR_WHITE,
                             FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])
        else:
            # Compute resize factor 'f'
            h, w, c = self._fer.input_image.shape
            h_c, w_c, c_c = self._input_container.shape
            h_ratio = h / h_c
            w_ratio = w / w_c
            if h_ratio > w_ratio:
                if h < (self._container_height * FERDemo._INPUT_IMAGE_SCALE_MIN):
                    f = (self._container_height * FERDemo._INPUT_IMAGE_SCALE_MIN) / float(h)
                else:
                    f = (self._container_height * FERDemo._INPUT_IMAGE_SCALE_MAX) / float(h)
            else:
                if w < (self._container_height * FERDemo._INPUT_IMAGE_SCALE_MIN):
                    f = (self._container_width * FERDemo._INPUT_IMAGE_SCALE_MIN) / float(w)
                else:
                    f = (self._container_width * FERDemo._INPUT_IMAGE_SCALE_MAX) / float(w)

            # Resize input image
            self._input_image = image_processing.resize(self._fer.input_image, f=f)

            # Set input image to the container
            h, w, c = self._input_image.shape
            x = int((self._container_height // 2) - (h // 2))
            y = int((self._container_width // 2) - (w // 2))

            self._input_container[x:(x + h), y:(y + w), :] = self._input_image

    def _draw_output_container(self, is_blank):
        self._output_container = self._get_container(0,
                                                     self._output_container_initial_position[1],
                                                     self._container_height,
                                                     self._container_width)

        if is_blank:
            image_processing.draw_text(self._output_container,
                             FERDemo._TEXT_BLANK_INPUT,
                             self._container_center_position - 60,
                             FERDemo._COLOUR_BGR_WHITE,
                             FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])
        else:
            if self._fer.face_image is None:
                image_processing.draw_text(self._output_container,
                                 FERDemo._TEXT_NO_FACE,
                                 self._container_center_position - 210,
                                 FERDemo._COLOUR_BGR_BLACK,
                                 FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                 FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])
            else:
                # Display ensemble and individual classifications
                # Resize face image
                face_image = image_processing.resize(self._fer.face_image, FERDemo._BLOCK_IMAGE_SIZE[self._screen_size])

                # Generate block of the ensemble prediction
                block = self._generate_block(FERDemo._TEXT_ENSEMBLE,
                                             self._fer.list_emotion[-1],
                                             self._fer.list_affect[-1][0],
                                             self._fer.list_affect[-1][1],
                                             face_image=face_image,
                                             x=0,
                                             y=self._output_container_initial_position[1])

                # Draw block ot the ensemble prediction
                image_processing.draw_image(self._output_container, block, (0, 0))

                # Branches
                for branch in range(len(self._fer.list_emotion) - 1):
                    # Superimpose saliency map on input face image
                    grad_cam = self._fer.get_grad_cam(branch)
                    if not (grad_cam is None):
                        grad_cam = image_processing.superimpose(grad_cam, face_image)

                    # Generate block of the branch prediction
                    block = self._generate_block(FERDemo._TEXT_BRANCH.format(branch + 1),
                                                 self._fer.list_emotion[branch],
                                                 self._fer.list_affect[branch][0],
                                                 self._fer.list_affect[branch][1],
                                                 grad_cam,
                                                 x=self._output_block_height * (branch + 1),
                                                 y=self._output_container_initial_position[1])

                    # Draw block of the branch prediction
                    image_processing.draw_image(self._output_container, block,
                                                (self._output_block_height * (branch + 1), 0))

    def _generate_block(self, network_name, emotion, valence, arousal, face_image=None, x=0, y=0):
        block = self._get_container(x, y, self._output_block_height, self._output_block_width)

        # Image
        if not (face_image is None):
            image_processing.draw_image(block, face_image, FERDemo._BLOCK_INIT_POS_IMAGE[self._screen_size])

        # Text: Ensemble
        image_processing.draw_text(block,
                                   network_name,
                                   FERDemo._BLOCK_INIT_POS_TEXT_NETWORK[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Emotion
        image_processing.draw_text(block,
                                   emotion,
                                   FERDemo._BLOCK_INIT_POS_TEXT_EMOTION[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Arousal
        image_processing.draw_text(block,
                                   FERDemo._TEXT_AROUSAL,
                                   FERDemo._BLOCK_INIT_POS_TEXT_AROUSAL[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Valence
        image_processing.draw_text(block,
                                   FERDemo._TEXT_VALENCE,
                                   FERDemo._BLOCK_INIT_POS_TEXT_VALENCE[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Bar: Arousal
        image_processing.draw_horizontal_bar(block,
                                             arousal,
                                             FERDemo._MAX_AROUSAL,
                                             FERDemo._BLOCK_INIT_POS_BAR_AROUSAL[self._screen_size],
                                             FERDemo._BLOCK_FINAL_POS_BAR_AROUSAL[self._screen_size],
                                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size],
                                             FERDemo._COLOUR_BGR_DARK_BLUE)

        # Bar: Valence
        image_processing.draw_horizontal_bar(block,
                                             np.abs(valence),
                                             FERDemo._MAX_VALENCE,
                                             FERDemo._BLOCK_INIT_POS_BAR_VALENCE[self._screen_size],
                                             FERDemo._BLOCK_FINAL_POS_BAR_VALENCE[self._screen_size],
                                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size],
                                             FERDemo._COLOUR_BGR_DARK_RED if valence < 0.0 else
                                             FERDemo._COLOUR_BGR_DARK_GREEN)

        return block

    def _generate_block_ensemble(self, network_name, emotion, valence, arousal, face_image=None, x=0, y=0):
        block = self._get_container(x, y, self._output_block_height_ensemble, self._output_block_width)

        # Image
        if not (face_image is None):
            image_processing.draw_image(block, face_image, FERDemo._BLOCK_INIT_POS_IMAGE_ENSEMBLE[self._screen_size])

        # Text: Ensemble
        image_processing.draw_text(block,
                                   network_name,
                                   FERDemo._BLOCK_INIT_POS_TEXT_NETWORK_ENSEMBLE[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Emotion
        image_processing.draw_text(block,
                                   emotion,
                                   FERDemo._BLOCK_INIT_POS_TEXT_EMOTION_ENSEMBLE[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Activation
        image_processing.draw_text(block,
                                   FERDemo._TEXT_ACTIVATION + "  {:.2f}".format(arousal),
                                   FERDemo._BLOCK_INIT_POS_TEXT_ACTIVATION[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Pleasant
        image_processing.draw_text(block,
                                   FERDemo._TEXT_PLEASANT + ("  0.00" if valence < 0 else "  {:.2f}".format(valence)),
                                   FERDemo._BLOCK_INIT_POS_TEXT_PLEASANT[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Text: Unpleasant
        image_processing.draw_text(block,
                                   FERDemo._TEXT_UNPLEASANT + ("  {:.2f}".format(valence) if valence < 0 else "  0.00"),
                                   FERDemo._BLOCK_INIT_POS_TEXT_UNPLEASANT[self._screen_size],
                                   FERDemo._COLOUR_BGR_BLACK,
                                   FERDemo._TEXT_PARAM_SCALE[self._screen_size],
                                   FERDemo._TEXT_PARAM_THICKNESS[self._screen_size])

        # Bar: Activation
        image_processing.draw_horizontal_bar(block,
                                             arousal,
                                             FERDemo._MAX_AROUSAL,
                                             FERDemo._BLOCK_INIT_POS_BAR_ACTIVATION[self._screen_size],
                                             FERDemo._BLOCK_FINAL_POS_BAR_ACTIVATION[self._screen_size],
                                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size],
                                             FERDemo._COLOUR_BGR_DARK_BLUE)

        # Bar: Pleasant
        image_processing.draw_horizontal_bar(block,
                                             0.0 if valence < 0.0 else valence,
                                             FERDemo._MAX_VALENCE,
                                             FERDemo._BLOCK_INIT_POS_BAR_PLEASANT[self._screen_size],
                                             FERDemo._BLOCK_FINAL_POS_BAR_PLEASANT[self._screen_size],
                                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size],
                                             FERDemo._COLOUR_BGR_DARK_GREEN)

        # Bar: Unpleasant
        image_processing.draw_horizontal_bar(block,
                                             np.abs(valence) if valence < 0.0 else 0.0,
                                             FERDemo._MAX_VALENCE,
                                             FERDemo._BLOCK_INIT_POS_BAR_UNPLEASANT[self._screen_size],
                                             FERDemo._BLOCK_FINAL_POS_BAR_UNPLEASANT[self._screen_size],
                                             FERDemo._TEXT_PARAM_THICKNESS[self._screen_size],
                                             FERDemo._COLOUR_BGR_DARK_RED)

        # Plot: Arousal and Valence
        if self._display_graph_ensemble:
            self._plot_arousal.append(arousal)
            self._plot_valence.append(valence)
            image_processing.draw_graph(block, self._plot_arousal, self._plot_valence,
                                        FERDemo._BLOCK_INIT_POS_GRAPH[self._screen_size],
                                        FERDemo._BLOCK_SAMPLE_GRAPH,
                                        FERDemo._TEXT_ACTIVATION_WITHOUT_TWO_DOTS,
                                        FERDemo._TEXT_PLEASANT_UNPLEASANT,
                                        FERDemo._COLOUR_BGR_BLUE,
                                        FERDemo._COLOUR_BGR_ORANGE,
                                        FERDemo._BLOCK_THICKNESS_GRAPH[self._screen_size],
                                        FERDemo._BLOCK_OFFSET_GRAPH[self._screen_size],
                                        FERDemo._BLOCK_FONT_SIZE_GRAPH[self._screen_size],
                                        FERDemo._COLOUR_BGR_DARK_GREY,
                                        FERDemo._BLOCK_SIZE_GRAPH[self._screen_size])
        return block

    def _draw_background(self):
        if (self._fer is None) or (self._fer.input_image is None):
            self._background = np.ones((self._height, self._width, 3), dtype=np.uint8) * FERDemo._COLOUR_G_DARK_GREY
        else:
            # Resize
            self._background = image_processing.resize(self._fer.input_image, f=np.maximum(
                np.maximum(self._fer.input_image.shape[0] / self._height, self._fer.input_image.shape[1] / self._width),
                np.maximum(self._height / self._fer.input_image.shape[0], self._width / self._fer.input_image.shape[1])
            ))[: self._height, :self._width, :]
            # Blur
            self._background = image_processing.blur(image_processing.blur(self._background, 40), 20)
            # Brightness
            mean = np.mean(self._background)
            gamma = 0.75 if mean > 100 else 1.5
            mean = mean if mean > 50 else 100
            self._background = np.clip((gamma * self._background) + mean, 0, 255).astype(np.uint8)

    def _get_container(self, x, y, h, w):
        return np.array(self._background[x:x+h, y:y+w, :])

    def update(self, fer):
        """
        Update screen.
        :param fer: (model.ml.fer.FER) An FER object.
        :return: void
        """
        self._fer = fer
        # Background
        self._draw_background()
        self._draw_input_container(self._fer is None)
        self._draw_output_container(self._fer is None)
        self._draw_screen()

    def show(self):
        cv2.imshow(self._window_name, self._screen)

    def is_running(self):
        return cv2.waitKey(1) != 27

    def quit(self):
        cv2.destroyWindow(self._window_name)
