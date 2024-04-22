# https://onnxruntime.ai/docs/get-started/with-python
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

# pip install onnxruntime-gpu
# pip install onnxruntime
# onnx is built into PyTorch: pip install torch
# pip install tf2onnx
# pipi nstall skl2onnx

# export the torch model using torch.onnx.export
torch.onnx.exprt(
    model,                              # model being run
    torch.randn(1, 28, 28).to(device),  # model input (or a tuple for multiple inputs)
    "fashion_mnist_model.onnx",         # where to save the model
    export_params=True,                 # store the trained parameter weights inside the model file
    # opset_version=10,                   # the onnx version to export the model to
    # do_constant_folding=True,           # whether to execue constant folding for optimization
    input_names=["input"],              # model's input names
    output_names=["output"],            # model's output names
    # dynamic_axes={                      # vatiable length axes
    #     "input": {0: "batch_size"},
    #     "output": {0: "batch_size"},
    # }
    )

# load the onnx model with onnx.load
import onnx

onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)    # model 유효성 검사

# create inference session using ort.InferenceSession
import numpy as np
import onnxruntime as ort

x, y = test_data[0][0], test_data[0][1]

ort_session = ort.InferenceSession(
    "fashion_mnist_model.onnx",
    providers = ["CPUExecutionProvider"]
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outputs = ort_session.run(None, ort_inputs)
# outputs = ort_session.run(None, {"input": x.numpy()})

# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


