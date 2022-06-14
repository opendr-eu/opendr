"""
Facial Expression Recognition (FER) class.

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""


class FER:
    """
    This class implements the facial expression recognition object that contains the elements
    to be displayed on the screen such as an input image and ESR-9's outputs.
    """

    def __init__(self, image=None, face_image=None, face_coordinates=None,
                 list_emotion=None, list_affect=None, list_grad_cam=None):
        """
        Initialize FER object.
        :param image: (ndarray) input image.
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
