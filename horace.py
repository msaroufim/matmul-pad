import torch
import time
import matplotlib.pyplot as plt
# Enable tensor cores
torch.set_float32_matmul_precision('high')
from tqdm import tqdm
import csv

f = open('output', 'w')
writer = csv.writer(f)
header = ["outer_size", "inner_size", "duration", "pad"]
writer.writerow(header)
cuda = torch.device('cuda')


def largest_closest_multiple(n, k=16):
    if n % k == 0:
        return n
    else:
        return n + k - (n % k)

durations_no_pad = []
durations_pad = []

for size1 in tqdm(range(10000, 10000 + 100)):
    size2 = size1
    tensor1 = torch.randn(size1,size2, device=cuda)
    tensor2 = torch.randn(size2, size1, device=cuda)

    tic = time.time()
    for _ in range(10):
        result = torch.matmul(tensor1, tensor2)
    torch.cuda.synchronize()

    toc = time.time()
    durations_no_pad.append(toc - tic)

    writer.writerow([str(size1), str(size2), str(toc - tic), "false"])

    tensor1 = torch.randn(largest_closest_multiple(size1),largest_closest_multiple(size2), device=cuda)
    tensor2 = torch.randn(largest_closest_multiple(size2), largest_closest_multiple(size1), device=cuda)

    tic = time.time()
    for _ in range(10):
        result = torch.matmul(tensor1, tensor2)
    torch.cuda.synchronize()
    toc = time.time()
    durations_pad.append(toc - tic)

    writer.writerow([str(largest_closest_multiple(size1)), str(largest_closest_multiple(size2)), str(toc - tic), "true"])



plt.plot(durations_no_pad[1:], label="no-pad")
plt.plot(durations_pad[1:], label="pad-to-16")
plt.ylabel('Time taken in seconds')
plt.xlabel('Dimension of Tensor size1, size2')
plt.legend()

plt.show()
plt.savefig('matmul-pad-nopad-big.png')
f.close()