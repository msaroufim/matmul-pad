import torch
import time
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

# Enable tensor cores
torch.set_float32_matmul_precision('high')
cuda = torch.device('cuda')

dtype_to_alignment_map = {torch.int8 : 16, torch.float16 : 8, torch.float32 : 4, torch.float64 : 2}

def largest_closest_multiple(n, k=16):
    if n % k == 0:
        return n
    else:
        return n + k - (n % k)

def matmul_test(size1, size2, dtype=torch.int8, pad=False):
    if pad:
        size1 = largest_closest_multiple(size1, dtype_to_alignment_map[dtype])
        size2 = largest_closest_multiple(size2, dtype_to_alignment_map[dtype])
    
    tensor1 = torch.randn(size1,size2, device=cuda)
    tensor2 = torch.randn(size2, size1, device=cuda)

    tensor1.to(dtype)
    tensor2.to(dtype)

    tic = time.time()
    result = torch.mm(tensor1, tensor2)
    torch.cuda.synchronize()
    toc = time.time()
    return toc - tic

def write_to_csv(file, data):
    writer = csv.writer(file)
    header = ["outer_size", "inner_size", "duration", "pad"]
    writer.writerow(header)
    for row in data:
        writer.writerow(row)

def run_tests(sizes, pad=False):
    durations = []
    data = []
    for size1 in tqdm(sizes):
        for size2 in tqdm(sizes):
            duration = matmul_test(size1, size2, pad)
            durations.append(duration)
            data.append([str(size1), str(size2), str(duration), str(pad)])
    return durations, data

def plot_results(durations_no_pad, durations_pad):
    plt.plot(durations_no_pad[1:], label="no-pad")
    plt.plot(durations_pad[1:], label="pad-to-16")
    plt.ylabel('Time taken in seconds')
    plt.xlabel('Dimension of Tensor size1, size2')
    plt.legend()

    plt.show()
    plt.savefig('matmul-pad-nopad-big.png')

def main():
    sizes = range(10000, 20000, 1000)
    durations_no_pad, data_no_pad = run_tests(sizes, False)
    durations_pad, data_pad = run_tests(sizes, True)

    with open('output', 'w') as f:
        write_to_csv(f, data_no_pad + data_pad)

    plot_results(durations_no_pad, durations_pad)

if __name__ == "__main__":
    main()