from torch import Tensor, nn


class ConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ELU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_reflection: bool = True,
    ) -> None:
        super().__init__()

        if use_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pad(x)
        out = self.conv(out)
        return out
