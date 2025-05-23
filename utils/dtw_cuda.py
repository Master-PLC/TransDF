# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------


##### from https://github.com/DYosplay/DsCGAN


import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

import pdb

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_dtw_cuda(D, bandwidth, max_i, max_j, inv_max_i, inv_max_j, n_passes, R, Ind):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            # if not (abs(i - j) > bandwidth > 0):
            if not (abs((i-1)*inv_max_i - (j-1)*inv_max_j) > bandwidth > 0):
                r0 = R[b, i - 1, j - 1] 
                r1 = R[b, i - 1, j] 
                r2 = R[b, i, j - 1]
                
                rmin = min(min(r0, r1), r2)
                if rmin == r0:
                    Ind[b, i, j] = 0 # Diagonal
                elif rmin == r1:
                    Ind[b, i, j] = 1 # Vertical
                else:
                    Ind[b, i, j] = 2 # Horizontal

                # if r0 <= r1 and r0 <= r2:
                #     Ind[b, i, j] = 0 # Diagonal
                #     rmin = r0
                # elif r1 <= r0 and r1 <= r2:
                #     Ind[b, i, j] = 1 # Vertical
                #     rmin = r1
                # else:
                #     Ind[b, i, j] = 2 # Horizontal
                #     rmin = r2
                
                R[b, i, j] = D[b, i - 1, j - 1] + rmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_dtw_backward_cuda(Ind, bandwidth, max_i, max_j, inv_max_i, inv_max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            # Don't compute if outside bandwidth
            # if not (abs(i - j) > bandwidth > 0):
            if not (abs((i-1)*inv_max_i - (j-1)*inv_max_j) > bandwidth > 0):
                a = 1 if Ind[k, i + 1, j] == 1 else 0
                b = 1 if Ind[k, i, j + 1] == 2 else 0
                c = 1 if Ind[k, i + 1, j + 1] == 0 else 0
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _DTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, bandwidth):
        dev = D.device
        dtype = D.dtype
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        bandwidth = max(bandwidth, max(1./(N-1), 1./(M-1)))
        bandwidth = torch.cuda.FloatTensor([bandwidth])        

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * np.inf #math.inf
        R[:, 0, 0] = 0
        Ind = torch.zeros((B, N + 2, M + 2), device=dev, dtype=torch.int32) 

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_dtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                               bandwidth.item(), N, M, 1./(N-1), 1./(M-1), n_passes,
                                               cuda.as_cuda_array(R), cuda.as_cuda_array(Ind))
        ctx.save_for_backward(Ind, bandwidth)
        # print(D)
        # print(R[:, 1:-1, 1:-1])
        return R[:, -2, -2], R.detach()

    @staticmethod
    def backward(ctx, grad_output, l):
        dev = grad_output.device
        dtype = grad_output.dtype
        Ind, bandwidth = ctx.saved_tensors

        B = Ind.shape[0]
        N = Ind.shape[1] - 2
        M = Ind.shape[2] - 2
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        Ind[:, :, -1] = 1 # Vertical
        Ind[:, -1, :] = 2 # Horizontal
        Ind[:, -1, -1] = 0 # Diagonal

        # Grid and block sizes are set same as done above for the forward() call
        compute_dtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(Ind),
                                                        bandwidth.item(), N, M, 1./(N-1), 1./(M-1), n_passes,
                                                        cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        # print(torch.mean(E, dim=[1, 2]))
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_dtw(D, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0  # For a proper value of R[:, 1, 1]!
    I = np.zeros((B, N + 2, M + 2), dtype=np.int32)
    ### Always do pruing even if bandwidth = 0
    bandwidth = max(bandwidth, max(1./(N-1), 1./(M-1)))
    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                # if 0 < bandwidth < np.abs(i - j):
                if 0 < bandwidth < np.abs((i-1)/(N-1) - (j-1)/(M-1)):
                    continue

                r0 = R[b, i - 1, j - 1]
                r1 = R[b, i - 1, j] 
                r2 = R[b, i, j - 1]
                
                rmin = min(min(r0, r1), r2)
                if rmin == r0:
                    I[b, i, j] = 0 # Diagonal
                elif rmin == r1:
                    I[b, i, j] = 1 # Vertical
                else:
                    I[b, i, j] = 2 # Horizontal

                # if r0 <= r1 and r0 <= r2:
                #     I[b, i, j] = 0 # Diagonal
                #     rmin = r0
                # elif r1 <= r0 and r1 <= r2:
                #     I[b, i, j] = 1 # Vertical
                #     rmin = r1
                # else:
                #     I[b, i, j] = 2 # Horizontal
                #     rmin = r2

                R[b, i, j] = D[b, i - 1, j - 1] + rmin
    return R, I

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_dtw_backward(I, bandwidth):
    B = I.shape[0]
    N = I.shape[1] - 2
    M = I.shape[2] - 2
    E = np.zeros((B, N + 2, M + 2))
    E[:, -1, -1] = 1
    I[:, :, -1] = 1 # Vertical, stop BP to the left regions. 
    I[:, -1, :] = 2 # Horizontal, stop BP to the above regions.
    I[:, -1, -1] = 0 # Diagonal, copy E[:,-1,-1] to E[:,-2,-2]
    ### Always do pruing even if bandwidth = 0
    bandwidth = max(bandwidth, max(1./(N-1), 1./(M-1)))
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                # Check the pruning condition
                # if 0 < bandwidth < np.abs(i - j):
                if 0 < bandwidth < np.abs((i-1.0)/(N-1) - (j-1.0)/(M-1)):
                    continue

                a = 1 if I[k, i + 1, j] == 1 else 0
                b = 1 if I[k, i, j + 1] == 2 else 0
                c = 1 if I[k, i + 1, j + 1] == 0 else 0

                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _DTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, bandwidth):
        dev = D.device
        dtype = D.dtype
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        b_ = bandwidth.item()
        R, I = torch.Tensor(compute_dtw(D_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(I, bandwidth)
        # print(D)
        # print(R[:, 1:-1, 1:-1])
        return R[:, -2, -2], R.detach()

    @staticmethod
    def backward(ctx, grad_output, l):
        dev = grad_output.device
        dtype = grad_output.dtype
        I, bandwidth = ctx.saved_tensors
        I_ = I.detach().cpu().numpy()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_dtw_backward(I_, b_)).to(dev).type(dtype)
        # print(torch.mean(E, dim=[1, 2]))
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None

# ----------------------------------------------------------------------------------------------------------------------
class DTW(torch.nn.Module):
    """
    The DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, normalize=False, bandwidth=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        """
        super(DTW, self).__init__()
        self.normalize = normalize
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda
        assert 0 <= self.bandwidth <= 1

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            # print("DTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
            use_cuda = False

        # Finally, return the correct function
        return _DTWCUDA.apply if use_cuda else _DTW.apply

    def _calc_distance_matrix(self, x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self._calc_distance_matrix(x, y)
            out = func_dtw(D, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return (out_xy - 1 / 2 * (out_xx + out_yy)) 
        else:
            D_xy = self._calc_distance_matrix(X, Y)
            return func_dtw(D_xy, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward
    start = timer()
    forward = sdtw(a, b)[0]
    print(forward)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = DTW(False, normalize=False, bandwidth=0.1)
    sdtw_cuda = DTW(True, normalize=False, bandwidth=0.1)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        assert torch.allclose(forward_cpu, forward_gpu.cpu())
        assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the scrip)
            times_cpu += [t_cpu]
            times_gpu += [t_gpu]
        
        # pdb.set_trace()

    # Average and log
    avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    print("\tCPU:     ", avg_cpu)
    print("\tGPU:     ", avg_gpu)
    print("\tSpeedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    # Not always pass the test... The warping paths may be different due to different numerical precisions on GPU & CPU.
    # E_GPU = np.load("../pytorch-softdtw-cuda-master/dtw_cuda_test/E_GPU.npy")
    # E_CPU = np.load("../pytorch-softdtw-cuda-master/dtw_cuda_test/E_CPU.npy")
    # D_GPU = np.load("../pytorch-softdtw-cuda-master/dtw_cuda_test/D_GPU.npy")
    # D_CPU = np.load("../pytorch-softdtw-cuda-master/dtw_cuda_test/D_CPU.npy")
    # print (np.where(E_GPU != E_CPU))
    # print(E_GPU[123:127, 119:122])
    # print(E_CPU[123:127, 119:122])
    # print(D_GPU[123:127, 119:122])
    # print(D_CPU[123:127, 119:122])

    profile(512, 17, 15, 64, tol_backward=1e-6)
    # profile(512, 64, 64, 2, tol_backward=1e-4)
    # profile(512, 256, 256, 2, tol_backward=1e-3) 

    # sdtw_cuda = DTW(True, normalize=False)
    # loss = []
    # for i in range(10):
    #     print(i)
    #     a_cpu = torch.randn((20, (i + 1) * 128, 2))
    #     b_cpu = torch.randn((20, (i + 1) * 128, 2))
    #     a_gpu = a_cpu.cuda()
    #     b_gpu = b_cpu.cuda()
    #     forward = sdtw_cuda(a_gpu, b_gpu)
    #     loss.append(torch.mean(forward).item() / ((i + 1)*128) )
    # from matplotlib import pyplot as plt
    # plt.plot(loss, "o") 
    # plt.show()