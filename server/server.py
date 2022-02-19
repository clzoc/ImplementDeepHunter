import numpy as np
from fastapi import FastAPI, WebSocket
import asyncio
import numba

@numba.njit
def assignAxis(input, para1, para2):
    for i in range(para1):
        for j in range(para2):
            input[i * 32 + j][0] = i
            input[i * 32 + j][1] = j

layoutNeu = np.zeros((64 * 32, 3), dtype=np.float32)
assignAxis(layoutNeu, 64, 32)

layoutInter = np.zeros((64 * 32, 3), dtype=np.int8)
assignAxis(layoutInter, 64, 32)

layoutLayerTop = np.zeros((64 * 32, 3), dtype=np.float32)
assignAxis(layoutLayerTop, 64, 32)

layoutCombi = np.zeros((64 * 32, 3), dtype=np.int8)
assignAxis(layoutCombi, 64, 32)

complement = np.zeros(32, dtype=np.int8)

arr1 = np.load("../message_to_frontend/neuron/allCoverageInformation.npy")
globalCoverage1 = np.ndarray.tolist(arr1[157])

arr2 = np.load("../message_to_frontend/internal/allCoverageInformation.npy")
globalCoverage2 = np.ndarray.tolist(arr2[157])

arr3 = np.load("../message_to_frontend/layerTop/allCoverageInformation.npy")
globalCoverage3 = np.ndarray.tolist(arr3[157])

arr4 = np.load("../message_to_frontend/combination/allCoverageInformation.npy")
globalCoverage4 = np.ndarray.tolist(arr4[157])

app = FastAPI()


@app.websocket_route("/nc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            neuInf = np.load(
                "../message_to_frontend/neuron/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr1[index])
            neuronNumber = neuInf[assist_iter].shape[0]
            layoutNeu[:, -1] = 0
            layoutNeu[0:neuronNumber, -1] = neuInf[assist_iter] * 20
            forsend = np.ndarray.tolist(layoutNeu)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage1[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            neuronNumber = neuInf[assist_iter].shape[0]
            layoutNeu[:, -1] = 0
            layoutNeu[0:neuronNumber, -1] = neuInf[assist_iter] * 20
            forsend = np.ndarray.tolist(layoutNeu)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage1[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/ic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            interInf = np.load(
                "../message_to_frontend/internal/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr2[index])
            neuronNumber = interInf[assist_iter].shape[0]
            layoutInter[:, -1] = 0
            layoutInter[0:neuronNumber, -1] = interInf[assist_iter]
            forsend = np.ndarray.tolist(layoutInter)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage2[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            neuronNumber = interInf[assist_iter].shape[0]
            layoutInter[:, -1] = 0
            layoutInter[0:neuronNumber, -1] = interInf[assist_iter]
            forsend = np.ndarray.tolist(layoutInter)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage2[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/lc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            layertopInf = np.load(
                "../message_to_frontend/layerTop/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr3[index])
            neuronNumber = layertopInf[assist_iter].shape[0]
            layoutLayerTop[:, -1] = 0
            layoutLayerTop[0:neuronNumber, -1] = 1 / (layertopInf[assist_iter] + 1)
            forsend = np.ndarray.tolist(layoutLayerTop)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage3[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            neuronNumber = layertopInf[assist_iter].shape[0]
            layoutLayerTop[:, -1] = 0
            layoutLayerTop[0:neuronNumber, -1] = 1 / (layertopInf[assist_iter] + 1)
            forsend = np.ndarray.tolist(layoutLayerTop)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage3[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc64")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = np.concatenate(
                (combiInf[assist_iter], complement), axis=0
            )
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = np.concatenate(
                (combiInf[assist_iter], complement), axis=0
            )
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc128")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = combiInf[assist_iter][0 : (64 * 32)]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = layoutCombi[:, -1] = combiInf[assist_iter][
                0 : (64 * 32)
            ]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc256")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = combiInf[assist_iter][0 : (64 * 32)]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = layoutCombi[:, -1] = combiInf[assist_iter][
                0 : (64 * 32)
            ]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc512")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = combiInf[assist_iter][0 : (64 * 32)]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = layoutCombi[:, -1] = combiInf[assist_iter][
                0 : (64 * 32)
            ]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc1024")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = combiInf[assist_iter][0 : (64 * 32)]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = layoutCombi[:, -1] = combiInf[assist_iter][
                0 : (64 * 32)
            ]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)


@app.websocket_route("/cc2048")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(websocket.state)
    assist_layer = 160
    assist_iter = 0
    while True:
        index = await websocket.receive_text()
        index = int(index)

        if index != assist_layer:
            assist_layer = index
            combiInf = np.load(
                "../message_to_frontend/combination/{}/inf.npy".format(assist_layer)
            )
            print(index)
            assist_iter = 0
            currentCoverage = np.ndarray.tolist(arr4[index])
            layoutCombi[:, -1] = combiInf[assist_iter][0 : (64 * 32)]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )
        else:
            assist_iter += 1
            layoutCombi[:, -1] = layoutCombi[:, -1] = combiInf[assist_iter][
                0 : (64 * 32)
            ]
            forsend = np.ndarray.tolist(layoutCombi)
            await websocket.send_json(
                [currentCoverage[assist_iter], globalCoverage4[assist_iter], forsend]
            )

        await asyncio.sleep(2)
