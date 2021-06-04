import warnings

import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor, nature_cnn


def post_concat_extractor(flat_observations, layers, type_str: str, act_fun=tf.tanh):
    latency = flat_observations
    for idx, layer_size in enumerate(layers):
        latency = act_fun(
            linear(latency, "post_concat_{}_fc{}".format(type_str, idx), layer_size, init_scale=np.sqrt(2)))
    return latency


def create_dual_extractor(num_direct_features,
                          cnn_custom_extractor=nature_cnn,
                          mlp_custom_extractor=mlp_extractor,
                          post_concat_custom_extractor=post_concat_extractor):
    """
    Create and return a function for augmented_nature_cnn
    used in stable-baselines.

    num_direct_features tells how many direct features there
    will be in the image.
    """

    def dual_extractor(input, mlp_net_arch, mlp_act_fun, **kwargs):
        """
        Copied from stable_baselines policies.py.
        This is nature CNN head where last channel of the image contains
        direct features.

        :param scaled_images: (TensorFlow Tensor) Image input placeholder
        :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
        :return: (TensorFlow Tensor) The CNN output layer
        @param post_concat_layers:
        """

        # Take last channel as direct features
        other_features = tf.contrib.slim.flatten(input[..., -1])
        # tf.print(other_features, [other_features], message="This is other_features: ")
        # Take known amount of direct features, rest are padding zeros
        other_features = other_features[:, :num_direct_features]
        tf.print(other_features, [other_features])
        images = input[..., :-1]
        with tf.Session() as sess:
            print([other_features])

        # Find CNN features
        img_output = cnn_custom_extractor(images, **kwargs)

        # Run MLP
        latent_policy, latent_value = mlp_custom_extractor(tf.layers.flatten(other_features),
                                                           mlp_net_arch,
                                                           mlp_act_fun)

        # Concat:
        policy_concat = tf.concat((img_output, latent_policy), axis=1)
        value_concat = tf.concat((img_output, latent_value), axis=1)

        # Post process
        latent_policy = post_concat_custom_extractor(tf.layers.flatten(policy_concat), [124], type_str="policy")
        latent_value = post_concat_custom_extractor(tf.layers.flatten(value_concat), [124], type_str="value")
        # Done
        return latent_policy, latent_value

    return dual_extractor


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, extractor=mlp_extractor, feature_extraction="mlp", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = extractor(self.processed_obs, **kwargs)
            else:  # Up to the user to make sure the extractor takes the correct input.
                pi_latent, vf_latent = extractor(self.processed_obs, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class MultiInputPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, extractor, reuse=False, **_kwargs):
        super(MultiInputPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                 extractor=extractor,
                                                 feature_extraction="mlp", **_kwargs)
