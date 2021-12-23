import torch

import torch_custom_lstm as custom

# test using own Pytorch model
model = custom.SliceLSTM([(3, 2), (2, 1)])

a = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])

b = model(a)

print("model:")
print(model)
print("")

print("input:")
print(a)
print("")

print("output:")
print(b)