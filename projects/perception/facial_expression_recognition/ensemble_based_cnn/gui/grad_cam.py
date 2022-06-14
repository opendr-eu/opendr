"""
Implementation of ESR-9 (Siqueira et al., 2020) trained on AffectNet (Mollahosseini et al., 2017) for emotion
and affect perception.

Reference:
    - Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
      Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
      (AAAI-20), pages 1–1, New York, USA.

    - Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. AffectNet: A database for facial expression, valence,
       and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), pp.18-31.

Adopted from:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

# External libraries
import torch
from torch.autograd import Variable


class GradCAM:
    """
    Implementation of the Grad-CAM visualization algorithm (Selvaraju et al., 2017).
    Generates saliency maps with respects to discrete emotion labels (the second last, fully-connected
    layer of ESR-9 (Siqueira et al., 2020)).

    Reference:
        Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017.
        Grad-cam: Visual explanations from deep networks via gradient-based localization.
        In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

        Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
        Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
        (AAAI-20), pages 1–1, New York, USA.
    """

    # def __init__(self, esr_base, esr_branch_to_last_conv_layer, esr_branch_from_last_conv_layer_to_emotion_output):
    def __init__(self, esr, device):
        self._zero_grad = esr.zero_grad
        self._esr_base = esr.base
        self._esr_branch_to_last_conv_layer = []
        self._esr_branch_from_last_conv_layer_to_emotion_output = []
        for branch in esr.convolutional_branches:
            self._esr_branch_to_last_conv_layer.append(branch.forward_to_last_conv_layer)
            self._esr_branch_from_last_conv_layer_to_emotion_output.append(
                branch.forward_from_last_conv_layer_to_output_layer)

        self._gradients = None
        self._device = device

    def __call__(self, x, i):
        # Clear gradients
        self._gradients = []

        # Forward activations to the last convolutional layer
        feature_maps = self._esr_base(x)
        feature_maps = self._esr_branch_to_last_conv_layer[i](feature_maps)

        # Saves gradients
        feature_maps.register_hook(self.set_gradients)

        # Forward feature maps to the discrete emotion output layer (the second last, fully-connected layer)
        output_activations = self._esr_branch_from_last_conv_layer_to_emotion_output[i](feature_maps)

        return feature_maps, output_activations

    def set_gradients(self, grads):
        self._gradients.append(grads)

    def get_mean_gradients(self):
        return self._gradients[0].mean(3).mean(2)[0]

    def grad_cam(self, x, list_y):
        list_saliency_maps = []

        for i, y in enumerate(list_y):
            # Set gradients to zero
            self._zero_grad()

            # Forward phase
            feature_maps, output_activations = self(x, i)
            feature_maps = feature_maps[0]

            # Backward the activation of the neuron associated to
            # the predicted emotion to the last convolutional layer
            one_hot = torch.zeros(output_activations.size())
            one_hot[0][y] = 1
            one_hot = Variable(one_hot, requires_grad=True).to(self._device)
            one_hot = torch.sum(one_hot * output_activations)

            # Back-propagate activations
            one_hot.backward(retain_graph=True)

            # Get mean gradient for every convolutional filter
            grad_cam_weights = self.get_mean_gradients()

            # Computes saliency map as a weighted sum of feature maps and mean gradient
            saliency_map = torch.zeros(feature_maps.size()[1:]).to(self._device)
            for i, w in enumerate(grad_cam_weights):
                saliency_map += w * feature_maps[i, :, :]

            # Normalize saliency maps
            saliency_map = torch.clamp(saliency_map, min=0)
            saliency_map -= torch.min(saliency_map)
            saliency_map /= torch.max(saliency_map)

            list_saliency_maps.append(saliency_map)

        # Return the list of normalized saliency maps
        return list_saliency_maps
