# pylint: skip-file


"""
FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2015-2018 Roger R Labbe Jr.

this is still experimental code, not intended for use.
"""


from __future__ import print_function, division
from random import normalvariate
from math import cos, sqrt, pi, exp
from filterpy.common import pretty_str


def gaussian_density(x, mu, sigma):
    return 1 / (sigma * sqrt(2 * pi)) * exp(-(x - mu)**2 / (2 * sigma**2))



class ProcessModel:
    def __init__(self, sigma_1=10., sigma_v=10., sigma_w=1.):
        self.sigma_1 = sigma_1
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w


    def sample_initial(self):
        return normalvariate(0, self.sigma_1)


    def initial_density(self, x):
        return gaussian_density(x, 0, self.sigma_1)


    def deterministic_transition(self, t, x):
        return x/2 + 25*x/(1+x**2) + 8*cos(1.2*t)
        #return x/2 + 25*x/(1+x**2)


    def sample_transition(self, t, x):
        return self.deterministic_transition(t, x) \
                + normalvariate(0, self.sigma_v)


    def transition_density(self, t, x, x_next):
        return gaussian_density(
            x_next,
            self.deterministic_transition(t, x), self.sigma_v)


    def sample_observation(self, t, x):
        return x**2/20 + normalvariate(0, self.sigma_w)


    def observation_density(self, t, y, x_sample):
        return gaussian_density(y, x_sample**2/20, self.sigma_w)


    def __repr__(self):
        return '\n'.join([
            'ProcessModel object',
            pretty_str('sigma_1', self.sigma_1),
            pretty_str('sigma_v', self.sigma_v),
            pretty_str('sigma_w', self.sigma_w)
            ])



def quad_data(n, interval):
    from scipy.special.orthogonal import p_roots

    a, b = interval

    # transform gauss quadrature data
    midpoint = (a+b)/2
    length = b-a
    orig_particles, orig_gauss_weights = p_roots(n)
    particles = [midpoint + x*length/2 for x in orig_particles]
    gauss_weights = length/2*orig_gauss_weights

    return particles, gauss_weights




def quadrature(model, n=100, max_time=20, interval=(-40, 40)):
    particles, gauss_weights = quad_data(n, interval)

    prob = [model.initial_density(x) for x in particles]
    assert abs(sum(p*w for p, w in zip(prob, gauss_weights))-1) < 1e-3
    prob_history = [prob]

    x = model.sample_initial()
    x_history = [x]

    for t in range(1, max_time):
        x = model.sample_transition(t, x)
        y = model.sample_observation(t, x)
        x_history.append(x)

        prediction_prob = [
            sum(model.transition_density(t, xj, xi) * prob[j]
                for j, xj in enumerate(particles))
            for xi in particles]

        prob = [model.observation_density(t, y, xi) *
                gauss_weights[i] *
                prediction_prob[i]
                for i, xi in enumerate(particles)]
        normalizer = sum(prob)
        prob = [p/normalizer for p in prob]
        prob_history.append(prob)

    return x_history, particles, prob_history



if __name__ == '__main__':

    import matplotlib.pyplot as plt


    pm = ProcessModel()
    xs = [1]

    for i in range(100):
        xs.append(pm.deterministic_transition(7., i))

    plt.plot(xs)


#x_history, particles, prob_history = quadrature(ProcessModel())
