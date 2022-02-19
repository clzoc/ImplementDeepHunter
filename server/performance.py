import asyncio
import os
import sys

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pathlib

import cv2
import numba
import numpy as np
import torch
import torch.nn as nn
import torchvision
from fastapi import FastAPI, WebSocket
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

func = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)


@numba.njit(parallel=True)
def initial(numpy_array):
    for fi in numba.prange(50):
        for se in numba.prange(64):
            for th in numba.prange(112):
                for fo in numba.prange(112):
                    numpy_array[fi][se][th * 112 + fo][0] = th
                    numpy_array[fi][se][th * 112 + fo][1] = fo

wait_send = np.zeros((50, 64, 112 * 112, 3), dtype=np.float32)

initial(wait_send)

@numba.njit(parallel=True)
def assign(inf, array):
    for fi in numba.prange(50):
        for se in numba.prange(64):
            for th in numba.prange(112):
                for fo in numba.prange(112):
                    # notice: for transpose, swap th and fo
                    array[fi][se][th * 112 + fo][2] = inf[fi][se][fo][th]

@numba.njit
def statistic(neuron, pairwise, coverage, abstract):
    for fi in range(50):
        idx = 0
        for se in range(63):
            for th in range(se + 1, 64):
                case = 2 * neuron[fi][se] + neuron[fi][th]
                pairwise[idx][case] = 1
                num = 0
                for k in range(4):
                    num += pairwise[idx][k]
                abstract[fi][se * 64 + th][2] = num
                idx += 1
        select = 0
        for i in range(32 * 63):
            count = 0
            for j in range(4): 
                count += pairwise[i][j]
            if count == 4:
                select += 1
        coverage[fi] = select / (32 * 63)

@numba.njit
def axis(abstract):
    for i in range(50):
        idx = 0
        for fi in range(64):
            for se in range(64):
                abstract[i][idx][0] = fi
                abstract[i][idx][1] = se
                idx += 1
    

                

with torch.no_grad():
    net = torchvision.models.resnet50(pretrained=True).eval().cuda()

    feature = []
    neuron_value = []
    handle = []

    def user_hook(module, input, output):
        # reduction = torch.mean(output, dim=3) # reduction = torch.mean(reduction, dim=2) # feature.append(reduction)
        feature.append(output)
        reduction = torch.mean(output, dim=3)
        reduction = torch.mean(reduction, dim=2)
        neuron_value.append(reduction)

    
    ancilla = 0
    for u in net.children():
        if type(u) == nn.modules.container.Sequential:
            for v in u.children():
                # type of every v is torchvision.models.resnet.Bottleneck
                for w in v.children():
                    if type(w) == nn.modules.container.Sequential:
                        for x in w.children():
                            if ancilla == 0:
                                handle.append(x.register_forward_hook(user_hook))
                                ancilla += 1
                            else:
                                ancilla += 1

                    else:
                        if ancilla == 0:
                            handle.append(w.register_forward_hook(user_hook))
                            ancilla += 1
                        else:
                            ancilla += 1

        elif type(u) != nn.modules.linear.Linear:
            if ancilla == 0:
                handle.append(u.register_forward_hook(hook=user_hook))
                ancilla += 1
            else:
                ancilla += 1

    dataset_path = "/hdd3/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"
    image_root = pathlib.Path(dataset_path)
    image_path = [str(x) for x in image_root.glob("*/*.JPEG")]

    bgr_numpy = [
        cv2.imread(image_path[0 * 50 + x]) for x in range(50)
    ]
    rgb_numpy = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in bgr_numpy]
    resize_numpy = [cv2.resize(x, (224, 224)) for x in rgb_numpy]
    norm_tensor = [func(x) for x in resize_numpy]
    input_tensor_cpu = torch.stack(norm_tensor, axis=0)
    input_tensor_gpu = input_tensor_cpu.cuda()

    output_tensor = net(input_tensor_gpu)
    del output_tensor

    information = feature[0].cpu().numpy()

    assign(information, wait_send)

    ready_send = np.ndarray.tolist(wait_send)

    neuron_inf = torch.gt(neuron_value[0], 0).type(torch.int8)
    neuron_inf = neuron_inf.cpu().numpy()
    pairwise_inf = np.zeros((32 * 63, 4), dtype=np.int8)
    coverage_inf = np.zeros(50, dtype=np.float32)

    abstract_inf = np.zeros((50, 64 * 64, 3), dtype=np.int8)
    axis(abstract_inf)

    statistic(neuron_inf, pairwise_inf, coverage_inf, abstract_inf)
    next_ready_send = np.ndarray.tolist(coverage_inf)

    third_ready_send = np.ndarray.tolist(abstract_inf)


    feature.clear()
    neuron_value.clear()
    for hook in handle:
        hook.remove()




app = FastAPI()

@app.websocket("/inf")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # data = await websocket.receive_text()
        # await websocket.send_text(f"Message text was: {data}")
        index = await websocket.receive_text()
        index = int(index)

        for iter in range(50):
            await websocket.send_json([third_ready_send[iter], next_ready_send[iter]])
            await asyncio.sleep(3)

@app.websocket("/snowball")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # data = await websocket.receive_text()
        # await websocket.send_text(f"Message text was: {data}")
        index = await websocket.receive_text()
        index = int(index)
        former = (index - 1) * 2 + 0
        latter = (index - 1) * 2 + 2

        for iter in range(50):
            await websocket.send_json([ready_send[iter][former:latter], next_ready_send[iter]])
            await asyncio.sleep(3)

# uvicorn.run(app="performance:app", host="59.78.194.133", port=8000, reload=True)



