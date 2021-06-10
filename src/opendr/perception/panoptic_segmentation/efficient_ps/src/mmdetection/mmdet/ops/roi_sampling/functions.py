import torch
import torch.autograd as autograd
from torch.autograd.function import once_differentiable

from . import _backend

_INTERPOLATION = {"bilinear": _backend.Interpolation.Bilinear, "nearest": _backend.Interpolation.Nearest}
_PADDING = {"zero": _backend.PaddingMode.Zero, "border": _backend.PaddingMode.Border}


class ROISampling(autograd.Function):
    @staticmethod
    def forward(ctx, x, bbx, idx, roi_size, interpolation, padding, valid_mask):
        ctx.save_for_backward(bbx, idx)
        ctx.input_shape = (x.size(0), x.size(2), x.size(3))
        ctx.valid_mask = valid_mask

        try:
            ctx.interpolation = _INTERPOLATION[interpolation]
        except KeyError:
            raise ValueError("Unknown interpolation {}".format(interpolation))
        try:
            ctx.padding = _PADDING[padding]
        except KeyError:
            raise ValueError("Unknown padding {}".format(padding))

        y, mask = _backend.roi_sampling_forward(x, bbx, idx, roi_size, ctx.interpolation, ctx.padding, valid_mask)

        if not torch.is_floating_point(x):
            ctx.mark_non_differentiable(y)
        if valid_mask:
            ctx.mark_non_differentiable(mask)
            return y, mask
        else:
            return y

    @staticmethod
    @once_differentiable
    def backward(ctx, *args):
        if ctx.valid_mask:
            dy, _ = args
        else:
            dy = args[0]

        assert torch.is_floating_point(dy), "ROISampling.backward is only defined for floating point types"
        bbx, idx = ctx.saved_tensors

        dx = _backend.roi_sampling_backward(dy, bbx, idx, ctx.input_shape, ctx.interpolation, ctx.padding)
        return dx, None, None, None, None, None, None


def roi_sampling(x, bbx, idx, roi_size, interpolation="bilinear", padding="border", valid_mask=False):
    """Sample ROIs from a batch of images using bi-linear interpolation

    ROIs are sampled from the input by bi-linear interpolation, using the following equations to transform from
    ROI coordinates to image coordinates:

        y_img = y0 + y_roi / h_roi * (y1 - y0),     for y_roi in range(0, h_roi)
        x_img = x0 + x_roi / w_roi * (x1 - x0),     for x_roi in range(0, w_roi)

    where `(h_roi, w_roi)` is the shape of the ROI and `(y0, x0, y1, x1)` are its bounding box coordinates on the image

    Parameters
    ----------
    x : torch.Tensor
        A tensor with shape N x C x H x W containing a batch of images to sample from
    bbx : torch.Tensor
        A tensor with shape K x 4 containing the bounding box coordinates of the ROIs in "corners" format
    idx : torch.Tensor
        A tensor with shape K containing the batch indices of the image each ROI should be sampled from
    roi_size : tuple of int
        The size `(h_roi, w_roi)` of the output ROIs
    interpolation : str
        Sampling mode, one of "bilinear" or "nearest"
    padding : str
        Padding mode, one of "border" or "zero"
    valid_mask : bool
        If `True` also return a mask tensor that indicates which points of the outputs where sampled from within the
        valid region of the input

    Returns
    -------
    y : torch.Tensor
        A tensor with shape K x C x h_roi x w_roi containing the sampled ROIs
    mask : torch.Tensor
        Optional output returned only when valid_mask is `True`: a mask tensor with shape K x h_roi x w_roi, whose
        entries are `!= 0` where the corresponding location in `y` was sampled from within the limits of the input image
    """
    return ROISampling.apply(x, bbx, idx, roi_size, interpolation, padding, valid_mask)

def invert_roi_bbx(bbx, roi_size, img_size):
    """Compute bbx coordinates to perform inverse roi sampling"""
    bbx = torch.stack([bbx[:,1], bbx[:,0], bbx[:,3], bbx[:, 2]], dim=1)
    bbx_size = bbx[:, 2:] - bbx[:, :2]
    return torch.cat([
        -bbx.new(roi_size) * bbx[:, :2] / bbx_size,
        bbx.new(roi_size) * (bbx.new(img_size) - bbx[:, :2]) / bbx_size
    ], dim=1)


__all__ = ['roi_sampling', 'invert_roi_bbx']

