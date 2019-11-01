import io
import numpy as np
import torch
from torch import nn
import torch.onnx
import torchvision.models as models

def main():
    model = models.resnext101_32x8d(pretrained=True)
    model.eval()

    batch_size = 32

    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # set export_params=True to export model params
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "resnext101_32x8d.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=False,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                     )

main()

