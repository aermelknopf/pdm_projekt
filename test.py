import torch
import torch_high_lvl_lstm as custom

slices = [(1, 1), (2, 2), (3, 3)]


slices_a = [x[0] for x in slices]
slices_b = [x[1] for x in slices]


for i in enumerate(zip(slices_a, slices_b)):
    print(i)