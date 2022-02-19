import pathlib
import time

import cv2
import numpy as np
import torch
import torchvision
import numba as nb
from numba import cuda as nbcu
from torch import nn
from torchvision import transforms

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start = time.time()

interData = []
interData.append(np.zeros((22, 1000, 64), dtype=np.int8))
interData.append(np.zeros((24, 1000, 128), dtype=np.int8))
interData.append(np.zeros((47, 1000, 256), dtype=np.int8))
interData.append(np.zeros((32, 1000, 512), dtype=np.int8))
interData.append(np.zeros((20, 1000, 1024), dtype=np.int8))
interData.append(np.zeros((12, 1000, 2048), dtype=np.int8))

@nbcu.jit
def update(batchSize, layersOut, neuronFile, neuronSize, ref):
    for img_idx in range(batchSize):
        gridThreadIdx = nbcu.threadIdx.x + nbcu.blockIdx.x * nbcu.blockDim.x
        layer = int(gridThreadIdx // neuronSize)
        ni = gridThreadIdx - layer * neuronSize
        activateValue = layersOut[layer][img_idx][ni]
        lowerBound = ref[layer][ni][0]
        upperBound = ref[layer][ni][1]
        interval = upperBound - lowerBound
        index = int(10 * (activateValue - lowerBound) / interval)
        neuronFile[layer][ni][index] = 1

# load the record file
neuron = []
neuron.append(np.load("./neuron_interval/neuronInterval64.npy"))
neuron.append(np.load("./neuron_interval/neuronInterval128.npy"))
neuron.append(np.load("./neuron_interval/neuronInterval256.npy"))
neuron.append(np.load("./neuron_interval/neuronInterval512.npy"))
neuron.append(np.load("./neuron_interval/neuronInterval1024.npy"))
neuron.append(np.load("./neuron_interval/neuronInterval2048.npy"))

neuronRef = []
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference64.npy"))
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference128.npy"))
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference256.npy"))
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference512.npy"))
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference1024.npy"))
neuronRef.append(np.load("../trust_project/neuron_interval_reference/neuronIntervalReference2048.npy"))

# convert the file to GPU memory
deviceNeuron = []
deviceNeuronInterval = []
for i in range(6):
    deviceNeuron.append(nbcu.to_device(neuron[i]))
    deviceNeuronInterval.append(nbcu.to_device(neuronRef[i]))

# indicate the data root and image path
data_root = pathlib.Path("/hdd3/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train")
image_path = [str(x) for x in data_root.glob("*/*.JPEG")]

# initiate classify activation information
feature1 = []
feature2 = []
feature3 = []
feature4 = []
feature5 = []
feature6 = []

# store the hook handle
handle = []

# load the model
network = torchvision.models.resnet50(pretrained=True).eval().cuda()

# hook function for classifying
def user_hook(module, input, output):
    reduction = torch.mean(output, dim=3)
    reduction = torch.mean(reduction, dim=2)
    neurons = reduction.shape[-1]
    if neurons == 64:
        feature1.append(reduction)
    elif neurons == 128:
        feature2.append(reduction)
    elif neurons == 256:
        feature3.append(reduction)
    elif neurons == 512:
        feature4.append(reduction)
    elif neurons == 1024:
        feature5.append(reduction)
    elif neurons == 2048:
        feature6.append(reduction)


# register the hook into the model
def register():
    for u in network.children():
        if type(u) == nn.modules.container.Sequential:
            for v in u.children():
                # type of every v is torchvision.models.resnet.Bottleneck
                for w in v.children():
                    if type(w) == nn.modules.container.Sequential:
                        for x in w.children():
                            handle.append(x.register_forward_hook(user_hook))

                    else:
                        handle.append(w.register_forward_hook(user_hook))

        elif type(u) != nn.modules.linear.Linear:
            handle.append(u.register_forward_hook(hook=user_hook))

register()

#define batchBlocks and batchSize, number of input images per iteration is batchBlocks x batchSize
batch_blocks = 1000
batch_size = 5

# normalize function
normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
# normalize = transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        normalization,
    ]
)

@nb.njit
def runsingle(file, para1, secondFile):
    for j in nb.prange(para1):
        secondFile[j] = np.count_nonzero(file[j]) 

index = []
index.append(np.load("./layerlocation/layerlocation64.npy"))
index.append(np.load("./layerlocation/layerlocation128.npy"))
index.append(np.load("./layerlocation/layerlocation256.npy"))
index.append(np.load("./layerlocation/layerlocation512.npy"))
index.append(np.load("./layerlocation/layerlocation1024.npy"))
index.append(np.load("./layerlocation/layerlocation2048.npy"))

with torch.no_grad():
    for batch_idx in range(batch_blocks):
        bgr_numpy = [
            cv2.imread(image_path[batch_idx * batch_size + x])
            for x in range(batch_size)
        ]
        rgb_numpy = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in bgr_numpy]
        resize_numpy = [cv2.resize(x, (224, 224)) for x in rgb_numpy]
        norm_numpy = [normalize(x) for x in resize_numpy]

        input_numpy = np.stack(norm_numpy, axis=0)
        input_tensor = torch.from_numpy(input_numpy).cuda()

        output_tensor = network(input_tensor)

        tensor1 = torch.stack(feature1, dim=0)
        tensor2 = torch.stack(feature2, dim=0)
        tensor3 = torch.stack(feature3, dim=0)
        tensor4 = torch.stack(feature4, dim=0)
        tensor5 = torch.stack(feature5, dim=0)
        tensor6 = torch.stack(feature6, dim=0)

        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = []
        feature5 = []
        feature6 = []

        update[22 * 2, 32](
            batch_size,
            tensor1,
            deviceNeuron[0],
            64,
            deviceNeuronInterval[0],
        )
        update[24 * 4, 32](
            batch_size,
            tensor2,
            deviceNeuron[1],
            128,
            deviceNeuronInterval[1],
        )
        update[47 * 8, 32](
            batch_size,
            tensor3,
            deviceNeuron[2],
            256,
            deviceNeuronInterval[2],
        )
        update[32 * 16, 32](
            batch_size,
            tensor4,
            deviceNeuron[3],
            512,
            deviceNeuronInterval[3],
        )
        update[20 * 32, 32](
            batch_size,
            tensor5,
            deviceNeuron[4],
            1024,
            deviceNeuronInterval[4],
        )
        update[12 * 64, 32](
            batch_size,
            tensor6,
            deviceNeuron[5],
            2048,
            deviceNeuronInterval[5],
        )

        nbcu.synchronize()

        for i in range(6):
            forCount = deviceNeuron[i].copy_to_host()
            for m in range(forCount.shape[0]):
                runsingle(forCount[m], forCount.shape[1], interData[i][m][batch_idx])

        print(batch_idx)

    for x in handle:
        x.remove()


for i in range(6):
    forCount = interData[i]
    for m in range(forCount.shape[0]):
        layer_index = index[i][m]
        np.save("../message_to_frontend/internal/{}/inf.npy".format(layer_index), interData[i][m]) 

print(time.time() - start)
