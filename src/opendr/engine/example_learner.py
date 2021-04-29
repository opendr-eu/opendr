# Copyright 2020 Aristotle University of Thessaloniki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from opendr.engine.learners import Learner


class ExampleLearner(Learner):
    # 1. The default values in constructor arguments can be set according to the algorithm.
    # 2. Some of the shared parameters, e.g. lr_schedule, backbone, etc., can be skipped here if not needed
    #    by the implementation.
    # 3. TODO Make sure the naming of the arguments is the same as the parent class arguments to keep it consistent
    #     for the end user.
    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='sgd', lr_schedule='',
                 backbone='default', network_head='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda', threshold=0.0, scale=1.0,
                 custom_param_1="default val", custom_param_2="default val"):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ExampleLearner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                             lr_schedule=lr_schedule, backbone=backbone, network_head=network_head,
                                             checkpoint_after_iter=checkpoint_after_iter,
                                             checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                             device=device, threshold=threshold, scale=scale)

        # Define the implementation specific parameters
        # TODO Make sure to do appropriate typechecks and provide valid default values for all custom parameters used.
        self.custom_param_1 = custom_param_1
        self.custom_param_2 = custom_param_2

    # All methods below are dummy implementations of the abstract methods that are inherited.
    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, params):
        pass

    def reset(self):
        pass

    def fit(self, dataset, val_dataset=None, logging_path='', silent=False, verbose=False):
        # The fit method's signature requires all of the above arguments to be present. The algorithm-specific
        # default values can be set here, e.g. set the default value of verbose to True
        pass

    def eval(self, dataset):
        pass

    def infer(self, batch, tracked_bounding_boxes=None):
        # In this infer dummy implementation, a custom argument is added as optional, so as not to change the basic
        # signature of the abstract method.
        # TODO The implementation must make sure it throws an appropriate error if the custom argument is needed and
        #  not provided (None).
        pass


# Overriding shared param through constructor, passing a value into a custom parameter
exampleLearner = ExampleLearner(lr=100.0, custom_param_1="custom value")
print("lr overridden through constructor   :", exampleLearner.lr)
print("batch_size default                  :", exampleLearner.batch_size)
# Overriding shared param after creation
exampleLearner.batch_size = 999
print("batch_size overridden after creation:", exampleLearner.batch_size)
# Printing custom parameters values
print("custom_param_1 overridden through constructor:", exampleLearner.custom_param_1)
print("custom_param_2                               :", exampleLearner.custom_param_2)
