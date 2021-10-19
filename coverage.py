import numpy as np
import numba as nb


@nb.njit
def run(file, para1, para2):
    out = 0
    for j in nb.prange(para1):
        for k in nb.prange(para2):
            if np.count_nonzero(file[j][k]) == 4:
                out += 1
    return out


total_number = 0
satisfied_number = 0

for i in range(6):
    pairwise = np.load(
        "./pairwise/pairwise{}.npy".format(pow(2, 6 + i)),
    )

    layer_number = pairwise.shape[0]
    layer_size = pairwise.shape[1]

    total_number += layer_number * layer_size

    satisfied_number += run(pairwise, layer_number, layer_size)


print("total number: ", total_number)
print("satisfied number: ", satisfied_number)
print("coverage: ", satisfied_number / total_number)
