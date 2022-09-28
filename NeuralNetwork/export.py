# %%
import os
import datetime
import base64
import struct
import torch
from network import NeuralNetwork
from parameters import INPUT_SIZE, OUTPUT_SIZE, TIMESTAMP
# %%
model_dir = "./data/models/"
model_name = f"{TIMESTAMP}.pth"
model_path = os.path.join(model_dir, model_name)
model = NeuralNetwork(INPUT_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(model_path))
# %%
model.state_dict()
# %%
def pack(value: float) -> bytes:
    return struct.pack('<d', value)
# %%
layer_names = ["in", "hidden1", "hidden2", "hidden3", "out"]
stream = b""

for suffix in layer_names:
    key = f"layer_{suffix}.weight"
    for v in model.state_dict()[key]:
        for v in v:
            stream += pack(v)

    key = f"layer_{suffix}.bias"
    for v in model.state_dict()[key]:
        stream += pack(v)
# %%
s = base64.b64encode(stream)
# %%
weight_name = f"{TIMESTAMP}.txt"
model_path = os.path.join("./data/weights", weight_name)
with open(model_path, mode="w") as f:
    f.write(s.decode("utf-8"))
