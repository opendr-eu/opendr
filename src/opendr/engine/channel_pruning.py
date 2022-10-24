import torch
from torch import nn


class ChannelPruningBase(nn.Module):

    collection = []

    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        body,
        weights,
        biases,
        input_dim: int,
        output_dim: int,
        total_dims: int,
        norm=2,
        targetable=True,
        pass_weights=True,
    ) -> None:

        super().__init__()

        self.body = body
        self.weights = weights
        self.biases = biases
        self.links = []
        self.rankings = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_dims = total_dims
        self.norm = norm
        self.targetable = targetable
        self.pass_weights = pass_weights

        if self.targetable:
            ChannelPruningBase.collection.append(self)

    def forward(self, x):

        if self.pass_weights:
            result = self.body(x, self.weights, self.biases)
        else:
            result = self.body(x)
        return result

    def add_link(self, x):
        self.links.append(x)

    def number_of_channels(self, is_input):
        dim = self.input_dim if is_input else self.output_dim
        result = self.weights.shape[dim]
        return result

    def compute_rankings(self, is_input):

        dim = self.input_dim if is_input else self.output_dim
        other_dims = [x for x in range(self.total_dims) if x != dim]
        self.rankings = torch.norm(self.weights, dim=other_dims, p=self.norm)

        return self.rankings

    def apply_pruning(self, is_input, num_to_prune):

        dim = self.input_dim if is_input else self.output_dim

        if self.rankings is None:
            self.compute_rankings(is_input)

        if len(self.rankings) <= num_to_prune:
            raise ValueError("Cannot prune more than we have")

        _, lowest_layer_ids = torch.topk(self.rankings, num_to_prune, largest=False)
        other_ids = [x for x in range(self.total_dims) if x not in lowest_layer_ids]
        all_index = [None if x != dim else other_ids for x in range(self.total_dims)]
        new_weights = self.weights[all_index]
        # old_weights = self.weights
        self.weights = new_weights
        # del old_weights

        self.rankings = None

        if not is_input:
            self.propagate_pruning(num_to_prune)

    def propagate_pruning(self, num_to_prune):
        for link in self.links:
            if not issubclass(link, ChannelPruningBase):
                raise ValueError(str(link) + " is not a Pruning layer")

            link.apply_pruning(True, num_to_prune)

    @staticmethod
    def collect():

        collection = ChannelPruningBase.collection
        ChannelPruningBase.collection = []

        return collection


class ChannelPruningConvolution2D(ChannelPruningBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        weights = nn.Parameter(
            torch.randn([
                    out_channels,
                    in_channels,
                    kernel_size[0],
                    kernel_size[1]
                ],
            ),
            requires_grad=True,
        )

        biases = None

        if bias:
            biases = nn.Parameter(
                torch.randn([
                        out_channels,
                    ],
                ),
                requires_grad=True,
            )

        def body(inputs, weights, biases):
            return nn.functional.conv2d(
                inputs,
                weights,
                biases,
                stride=stride,
                **kwargs
            )

        super().build(
            body,
            weights,
            biases,
            input_dim=1,
            output_dim=0,
            total_dims=4,
        )


class ChannelPruningLinear(ChannelPruningBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        body = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        super().build(
            body,
        )


class ChannelPruningConvolutionTranspose2D(ChannelPruningBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        weights = nn.Parameter(
            torch.randn([
                    in_channels,
                    out_channels,
                    kernel_size[0],
                    kernel_size[1]
                ],
            ),
            requires_grad=True,
        )

        biases = None

        if bias:
            biases = nn.Parameter(
                torch.randn([
                        out_channels,
                    ],
                ),
                requires_grad=True,
            )

        def body(inputs, weights, biases):
            return nn.functional.conv_transpose2d(
                inputs,
                weights,
                biases,
                stride=stride,
                **kwargs
            )

        super().build(
            body,
            weights,
            biases,
            input_dim=0,
            output_dim=1,
            total_dims=4,
        )


class ChannelPruningBatchNorm2D(ChannelPruningBase):
    def __init__(
        self,
        in_features: int,
        eps=1e-3,
        momentum=0.01,
        **kwargs,
    ) -> None:

        super().__init__()

        self.in_features = in_features
        self.eps = eps
        self.momentum = momentum
        self.kwargs = kwargs

        body = nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, **kwargs)

        super().build(
            body,
            None,
            None,
            input_dim=None,
            output_dim=None,
            total_dims=None,
            targetable=False,
            pass_weights=False,
        )

    def compute_rankings(self, _):
        pass

    def apply_pruning(self, _, num_to_prune):

        if self.in_features <= num_to_prune:
            raise ValueError("Cannot prune more than we have")

        self.in_features -= num_to_prune

        self.body = nn.BatchNorm2d(self.in_features, eps=self.eps, momentum=self.momentum, **self.kwargs)

        self.propagate_pruning(num_to_prune)

    def propagate_pruning(self, num_to_prune):
        for link in self.links:
            if not issubclass(link, ChannelPruningBase):
                raise ValueError(str(link) + " is not a Pruning layer")

            link.apply_pruning(True, num_to_prune)
