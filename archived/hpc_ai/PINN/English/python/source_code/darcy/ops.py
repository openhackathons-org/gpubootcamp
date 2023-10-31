import torch
import torch.nn.functional as F


def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                -0.5,
                0.0,
                0.5,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def ddx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                1.0,
                -2.0,
                1.0,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx ** 2) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output