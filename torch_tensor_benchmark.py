# testscript to benchmark runtime of pytorch tensor operations
import torch
import time


def multi_cat(num_tensors=100, shape=(20, 2)):
    result = torch.rand(shape, dtype=torch.float32)


    for i in range(num_tensors - 1):
        new_tensor = torch.rand(shape, dtype=torch.float32)
        result = torch.cat((result, new_tensor), dim=1)

    return result


def list_into_cat(num_tensors=100, shape=(20,2)):
    acc = []

    for i in range(num_tensors):
        new_tensor = torch.rand(shape, dtype=torch.float32)
        acc.append(new_tensor)

    return torch.cat(acc, dim=1)


def large_tensor_slicewrite(num_tensors=100, shape=(20, 2)):
    result = torch.empty((shape[0], num_tensors * shape[1]), dtype=torch.float32)

    start_col = 0

    for i in range(num_tensors):
        end_col = start_col + shape[1]
        result[:, start_col:end_col] = torch.rand(shape, dtype=torch.float32)
        start_col += shape[1]

    return result


def time_fn(fn):
    start_time = time.time()
    fn
    end_time = time.time()
    return end_time - start_time


if __name__ == '__main__':

    tensor_shape = (20, 2)
    num_tensors = 1000000

    print(time_fn(list_into_cat(num_tensors=num_tensors, shape=tensor_shape)))
    print(time_fn(large_tensor_slicewrite(num_tensors=num_tensors, shape=tensor_shape)))

    print(large_tensor_slicewrite(num_tensors=num_tensors, shape=tensor_shape))

