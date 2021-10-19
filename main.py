import pathlib
import time

import cv2
import numpy as np
import torch
import torchvision
from numba import cuda as nbcu
from torch import nn
from torchvision import transforms


class Information:
    def __init__(self, path1, path2) -> None:
        self.index = []
        self.index.append(np.load(path1 + "/index64.npy"))
        self.index.append(np.load(path1 + "/index128.npy"))
        self.index.append(np.load(path1 + "/index256.npy"))
        self.index.append(np.load(path1 + "/index512.npy"))
        self.index.append(np.load(path1 + "/index1024.npy"))
        self.index.append(np.load(path1 + "/index2048.npy"))

        self.device_index = []
        self.pairwise = []
        self.pairwise.append(np.load(path2 + "/pairwise64.npy"))
        self.pairwise.append(np.load(path2 + "/pairwise128.npy"))
        self.pairwise.append(np.load(path2 + "/pairwise256.npy"))
        self.pairwise.append(np.load(path2 + "/pairwise512.npy"))
        self.pairwise.append(np.load(path2 + "/pairwise1024.npy"))
        self.pairwise.append(np.load(path2 + "/pairwise2048.npy"))

        self.device_pairwise = []

    def convert_to_device(self):
        for i in range(6):
            self.device_index.append(nbcu.to_device(self.index[i]))

        for i in range(6):
            self.device_pairwise.append(nbcu.to_device(self.pairwise[i]))


class Process:
    def __init__(self, path) -> None:
        image_root = pathlib.Path(path)
        self.image_path = [str(x) for x in image_root.glob("*/*.JPEG")]

        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # normalize = transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                normalization,
            ]
        )


class LayerActivation:
    def __init__(self, network) -> None:
        self.feature1 = []
        self.feature2 = []
        self.feature3 = []
        self.feature4 = []
        self.feature5 = []
        self.feature6 = []

        self.handle = []
        self.net = network

    def user_hook(self, module, input, output):
        reduction = torch.mean(output, dim=3)
        reduction = torch.mean(reduction, dim=2)
        neurons = reduction.shape[-1]
        if neurons == 64:
            self.feature1.append(reduction)
        elif neurons == 128:
            self.feature2.append(reduction)
        elif neurons == 256:
            self.feature3.append(reduction)
        elif neurons == 512:
            self.feature4.append(reduction)
        elif neurons == 1024:
            self.feature5.append(reduction)
        elif neurons == 2048:
            self.feature6.append(reduction)

    def register(self):
        for u in self.net.children():
            if type(u) == nn.modules.container.Sequential:
                for v in u.children():
                    # type of every v is torchvision.models.resnet.Bottleneck
                    for w in v.children():
                        if type(w) == nn.modules.container.Sequential:
                            for x in w.children():
                                self.handle.append(
                                    x.register_forward_hook(self.user_hook)
                                )

                        else:
                            self.handle.append(w.register_forward_hook(self.user_hook))

            elif type(u) != nn.modules.linear.Linear:
                self.handle.append(u.register_forward_hook(hook=self.user_hook))

    def clean(self):
        self.feature1 = []
        self.feature2 = []
        self.feature3 = []
        self.feature4 = []
        self.feature5 = []
        self.feature6 = []

    def remove(self):
        for x in self.handle:
            x.remove()


class Grid:
    def __init__(self) -> None:
        self.blockspergrid = []
        self.threadsperblock = []
        self.pairwise_size = []

    def assignment(self):
        self.blockspergrid.append(63 * 22)
        self.blockspergrid.append(254 * 24)
        self.blockspergrid.append(1020 * 47)
        self.blockspergrid.append(511 * 32)
        self.blockspergrid.append(2046 * 20)
        self.blockspergrid.append(8188 * 12)

        self.threadsperblock.append(32)
        self.threadsperblock.append(32)
        self.threadsperblock.append(32)
        self.threadsperblock.append(256)
        self.threadsperblock.append(256)
        self.threadsperblock.append(256)

        self.pairwise_size.append(2016)
        self.pairwise_size.append(8128)
        self.pairwise_size.append(32640)
        self.pairwise_size.append(130816)
        self.pairwise_size.append(523776)
        self.pairwise_size.append(2096128)


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


if __name__ == "__main__":
    start = time.time()

    index_path = "./index"
    pairwise_path = "./pairwise"
    inf = Information(index_path, pairwise_path)
    inf.convert_to_device()

    dataset_path = "/hdd3/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"
    process = Process(dataset_path)

    grid = Grid()
    grid.assignment()

    network = torchvision.models.resnet50(pretrained=True).eval().cuda()
    internal = LayerActivation(network)
    internal.register()

    batch_blocks = 100
    batch_size = 50

    with torch.no_grad():

        for batch_idx in range(batch_blocks):
            bgr_numpy = [
                cv2.imread(process.image_path[batch_idx * batch_size + x])
                for x in range(batch_size)
            ]
            rgb_numpy = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in bgr_numpy]
            resize_numpy = [cv2.resize(x, (224, 224)) for x in rgb_numpy]
            norm_numpy = [process.normalize(x) for x in resize_numpy]

            input_numpy = np.stack(norm_numpy, axis=0)
            input_tensor = torch.from_numpy(input_numpy).cuda()

            output_tensor = internal.net(input_tensor)

            tensor1 = torch.stack(internal.feature1, dim=0)
            tensor2 = torch.stack(internal.feature2, dim=0)
            tensor3 = torch.stack(internal.feature3, dim=0)
            tensor4 = torch.stack(internal.feature4, dim=0)
            tensor5 = torch.stack(internal.feature5, dim=0)
            tensor6 = torch.stack(internal.feature6, dim=0)

            tensor1 = torch.gt(tensor1, 0).type(torch.int8)
            tensor2 = torch.gt(tensor2, 0).type(torch.int8)
            tensor3 = torch.gt(tensor3, 0).type(torch.int8)
            tensor4 = torch.gt(tensor4, 0).type(torch.int8)
            tensor5 = torch.gt(tensor5, 0).type(torch.int8)
            tensor6 = torch.gt(tensor6, 0).type(torch.int8)

            internal.clean()

            update[grid.blockspergrid[0], grid.threadsperblock[0]](
                batch_size,
                tensor1,
                inf.device_index[0],
                inf.device_pairwise[0],
                grid.pairwise_size[0],
            )
            update[grid.blockspergrid[1], grid.threadsperblock[1]](
                batch_size,
                tensor2,
                inf.device_index[1],
                inf.device_pairwise[1],
                grid.pairwise_size[1],
            )
            update[grid.blockspergrid[2], grid.threadsperblock[2]](
                batch_size,
                tensor3,
                inf.device_index[2],
                inf.device_pairwise[2],
                grid.pairwise_size[2],
            )
            update[grid.blockspergrid[3], grid.threadsperblock[3]](
                batch_size,
                tensor4,
                inf.device_index[3],
                inf.device_pairwise[3],
                grid.pairwise_size[3],
            )
            update[grid.blockspergrid[4], grid.threadsperblock[4]](
                batch_size,
                tensor5,
                inf.device_index[4],
                inf.device_pairwise[4],
                grid.pairwise_size[4],
            )
            update[grid.blockspergrid[5], grid.threadsperblock[5]](
                batch_size,
                tensor6,
                inf.device_index[5],
                inf.device_pairwise[5],
                grid.pairwise_size[5],
            )

            nbcu.synchronize()

        internal.remove()

    for i in range(6):
        np.save(
            "./pairwise/pairwise{}.npy".format(pow(2, 6 + i)),
            inf.device_pairwise[i].copy_to_host(),
        )

    print(time.time() - start)
