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

combiData = []
combiData.append(np.zeros((22, 1000, 63 * 32), dtype=np.int8))
combiData.append(np.zeros((24, 1000, 127 * 64), dtype=np.int8))
combiData.append(np.zeros((47, 1000, 255 * 128), dtype=np.int8))
combiData.append(np.zeros((32, 1000, 511 * 256), dtype=np.int8))
combiData.append(np.zeros((20, 1000, 1023 * 512), dtype=np.int8))
combiData.append(np.zeros((12, 1000, 2047 * 1024), dtype=np.int8))

@nbcu.jit
def update(batch_size, layers_out, index, pairwise, pair_size):
    for img_idx in range(batch_size):
        gridThreadIdx = nbcu.threadIdx.x + nbcu.blockIdx.x * nbcu.blockDim.x
        layer = int(gridThreadIdx // pair_size)
        pair = gridThreadIdx - layer * pair_size
        fni = index[pair][0]
        sni = index[pair][1]
        case = 2 * layers_out[layer][img_idx][fni] + layers_out[layer][img_idx][sni]
        pairwise[layer][pair][case] = 1

pre_index = []
pre_index.append(np.load("../trust_project/index/index64.npy"))
pre_index.append(np.load("../trust_project/index/index128.npy"))
pre_index.append(np.load("../trust_project/index/index256.npy"))
pre_index.append(np.load("../trust_project/index/index512.npy"))
pre_index.append(np.load("../trust_project/index/index1024.npy"))
pre_index.append(np.load("../trust_project/index/index2048.npy"))

neuron = []
neuron.append(np.load("./pairwise/pairwise64.npy"))
neuron.append(np.load("./pairwise/pairwise128.npy"))
neuron.append(np.load("./pairwise/pairwise256.npy"))
neuron.append(np.load("./pairwise/pairwise512.npy"))
neuron.append(np.load("./pairwise/pairwise1024.npy"))
neuron.append(np.load("./pairwise/pairwise2048.npy"))

# convert the file to GPU memory
devicePreIndex = []
deviceNeuron = []
for i in range(6):
    devicePreIndex.append(nbcu.to_device(pre_index[i]))
    deviceNeuron.append(nbcu.to_device(neuron[i]))

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
batch_size = 50

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

        tensor1 = torch.gt(tensor1, 0).type(torch.int8)
        tensor2 = torch.gt(tensor2, 0).type(torch.int8)
        tensor3 = torch.gt(tensor3, 0).type(torch.int8)
        tensor4 = torch.gt(tensor4, 0).type(torch.int8)
        tensor5 = torch.gt(tensor5, 0).type(torch.int8)
        tensor6 = torch.gt(tensor6, 0).type(torch.int8)

        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = []
        feature5 = []
        feature6 = []

        update[22 * 63, 32](
            batch_size,
            tensor1,
            devicePreIndex[0],
            deviceNeuron[0],
            2016,
        )
        update[24 * 254, 32](
            batch_size,
            tensor2,
            devicePreIndex[1],
            deviceNeuron[1],
            8128,
        )
        update[47 * 1020, 32](
            batch_size,
            tensor3,
            devicePreIndex[2],
            deviceNeuron[2],
            32640,
        )
        update[32 * 511, 256](
            batch_size,
            tensor4,
            devicePreIndex[3],
            deviceNeuron[3],
            130816,
        )
        update[20 * 2046, 256](
            batch_size,
            tensor5,
            devicePreIndex[4],
            deviceNeuron[4],
            523776,
        )
        update[12 * 8188, 256](
            batch_size,
            tensor6,
            devicePreIndex[5],
            deviceNeuron[5],
            2096128,
        )

        nbcu.synchronize()

        for i in range(6):
            forCount = deviceNeuron[i].copy_to_host()
            for m in range(forCount.shape[0]):
                runsingle(forCount[m], forCount.shape[1], combiData[i][m][batch_idx])

        print(batch_idx)

    for x in handle:
        x.remove()

for i in range(6):
    forCount = combiData[i]
    for m in range(forCount.shape[0]):
        layer_index = index[i][m]
        np.save("../message_to_frontend/combination/{}/inf.npy".format(layer_index), combiData[i][m]) 

print(time.time() - start)

