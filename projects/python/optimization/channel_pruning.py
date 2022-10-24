from multiprocessing.sharedctypes import Value
from typing import Any, Callable, List
import torch
from torch import nn

from opendr.engine.channel_pruning import ChannelPruningBase
from opendr.engine.learners import Learner


def default_pruning_probability_per_step(n):
    return 0.2 - 0.02 * n


class ChannelPruner:
    def __init__(
        self,
        learner: Learner,
        collection: List[ChannelPruningBase],
        pruning_probability_per_step=default_pruning_probability_per_step,
        ranking="local",  # local, global
    ) -> None:
        self.collection = collection
        self.pruning_probability_per_step = pruning_probability_per_step
        self.ranking = ranking
        self.learner = learner
        self.steps = 0

    def prune(self, steps, eval, fine_tune, save):

        results = []

        for i in range(steps):

            if self.ranking == "local":
                self.step_local()
            elif self.ranking == "local":
                self.step_global()
            else:
                raise ValueError("ranking should be local or global")
            fine_tune(self.learner, i)
            results.append(eval(self.learner))
            save(self.learner, i)

        return results

    def step_global(self):

        raise NotImplementedError()

        pruning_probability = self.pruning_probability_per_step(self.steps)

        rankings = [a.compute_rankings() for a in self.collection]

        self.steps += 1

    def step_local(self):

        pruning_probability = self.pruning_probability_per_step(self.steps)
        to_prune = [
            int(pruning_probability * a.number_of_channels(False))
            for a in self.collection
        ]

        for layer in self.collection:
            layer.apply_pruning(False, to_prune)

        self.steps += 1


def prune_learner(
    learner: Learner,
    steps: int,
    eval: Callable[[Learner], Any],
    fine_tune: Callable[[Learner, int], None],
    save: Callable[[Learner, int], None],
    ranking: str,
    pruning_probability_per_step=default_pruning_probability_per_step,
):

    collection = ChannelPruningBase.collect()

    if len(collection) <= 0:
        raise ValueError("No ChannelPruning layers found")

    pruner = ChannelPruner(
        learner,
        collection,
        pruning_probability_per_step=pruning_probability_per_step,
        ranking=ranking,
    )

    pruner.prune(steps, eval, fine_tune, save)
