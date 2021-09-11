
#define EIGEN_USE_GPU
#include "field_select_kernel.h"
#include <cuda.h>
#include <stdint.h>

__global__ void
target_field_mask(unsigned int const nx,
                  unsigned int const ny,
                  unsigned int const nxy,
                  long long int const* target_field,
                  long long int const* field_ids,
                  int* counts,
                  int* conds,
                  int* mask)
{
    extern __shared__ int local_results[];
    long long int idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long int idy = blockDim.y * blockIdx.y + threadIdx.y;
    long long int tid = nx * idy + idx;

    if (tid >= nxy || idy >= ny || idx >= nx) {
        return;
    }

    // 计算mask
    mask[tid] = field_ids[tid] == target_field[0];
    local_results[nx * threadIdx.y + threadIdx.x] = mask[tid];
    __syncthreads();

    atomicAdd(counts, mask[tid]); //统计个数

    tid = nx * threadIdx.y + threadIdx.x;

    // reduce mask
    for (long long int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_results[tid] = max(local_results[tid], local_results[tid + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(&conds[idy], local_results[tid]);
    }
}

__global__ void
where(unsigned int const nx,
      unsigned int const ny,
      unsigned int const nxy,
      int const* mask,
      long long int const* feat_ids,
      float const* feat_values,
      int const* conds,
      int* counts,
      long long int* output_indices)
{
    unsigned int const idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int const idy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int const tid = nx * idy + idx;

    if (tid >= nxy || idy >= ny || idx >= nx) {
        return;
    }

    if (mask[tid]) {
        int n = atomicSub(counts, 1);
        output_indices[2 * n] = (long long int)idy;
        output_indices[2 * n + 1] = (long long int)idx;
    }

    if (idx == 0 && conds[idy] == 0) {
        int n = atomicSub(counts, 1);
        output_indices[2 * n] = (long long int)idy;
        output_indices[2 * n + 1] = (long long int)idx;
    }
}

__global__ void
select_feat(unsigned int const nx,
            unsigned int const ny,
            unsigned int const nxy,
            long long int const counts,
            int const* conds,
            long long int const* indices,
            long long int const* feat_ids,
            float const* feat_values,
            long long int* ids,
            float* values)
{
    auto const idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto const idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto const tid = idy + idx;

    if (tid > counts) {
        return;
    }

    auto const y = indices[tid * 2];
    auto const x = indices[tid * 2 + 1];

    if (conds[y] == 0) {
        ids[tid] = 0;
        values[tid] = 0;
    } else {
        ids[tid] = feat_ids[nx * y + x];
        values[tid] = feat_values[nx * y + x];
    }
}
FieldSelectFuctor<Eigen::GpuDevice>::FieldSelectFuctor(Eigen::GpuDevice const& device)
    : device_(device)
{}

template<typename Func, typename... Args>
static void
luanch_cuda_kernel(Func kernel,
                   cudaStream_t const& stream,
                   unsigned int const nx,
                   unsigned int const ny,
                   unsigned int const nxy,
                   int smem,
                   Args&&... args)
{
    int blocksize = 0;
    int gridsize = 0;

    cudaOccupancyMaxPotentialBlockSize(&gridsize, &blocksize, kernel);
    dim3 block(blocksize, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    if (smem < 0) {
        smem = blocksize * sizeof(int);
    }

    kernel<<<grid, block, smem, stream>>>(nx, ny, nxy, std::forward<Args>(args)...);
}

void
FieldSelectFuctor<Eigen::GpuDevice>::launch_target_field_mask(const unsigned int nx,
                                                              const unsigned int ny,
                                                              const unsigned int nxy,
                                                              const long long int* target_field,
                                                              const long long int* field_ids,
                                                              int* counts,
                                                              int* conds,
                                                              int* mask,
                                                              int& cpu_counts)
{
    cpu_counts = 0;
    int* cpu_counts_portable = nullptr;
    int* cpu_conds = nullptr;
    cudaHostAlloc(&cpu_conds, sizeof(int) * ny, cudaHostAllocDefault);
    cudaHostAlloc(&cpu_counts_portable, sizeof(int), cudaHostAllocDefault);

    assert(cpu_conds);
    assert(cpu_counts_portable);

    auto const& stream = device_.stream();
    cudaMemsetAsync(counts, 0, sizeof(int), stream);
    cudaMemsetAsync(conds, 0, ny * sizeof(int), stream);
    luanch_cuda_kernel(
        target_field_mask, stream, nx, ny, nxy, -1, target_field, field_ids, counts, conds, mask);
    cudaMemcpyAsync(cpu_counts_portable, counts, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpu_conds, conds, sizeof(int) * ny, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // wait for done

    cpu_counts =
        std::accumulate(cpu_conds, cpu_conds + ny, *cpu_counts_portable, [](int const lhs, int const rhs) {
            return lhs + !rhs;
        });

    *cpu_counts_portable = cpu_counts - 1;
    cudaMemcpyAsync(counts, cpu_counts_portable, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream); // wait for done
    cudaFreeHost(cpu_conds);
    cudaFreeHost(cpu_counts_portable);
}

void
FieldSelectFuctor<Eigen::GpuDevice>::luanch_select_feat(const unsigned int nx,
                                                        const unsigned int ny,
                                                        const unsigned int nxy,
                                                        const int* mask,
                                                        const long long int* feat_ids,
                                                        const float* feat_values,
                                                        const int* conds,
                                                        int* counts,
                                                        long long int* output_indices,
                                                        long long int* output_feat_ids,
                                                        float* output_feat_values)
{
    auto const& stream = device_.stream();
    int* cpu_counts = nullptr;
    cudaHostAlloc(&cpu_counts, sizeof(int), cudaHostAllocDefault);
    assert(cpu_counts);
    cudaMemcpyAsync(cpu_counts, counts, sizeof(int), cudaMemcpyDeviceToHost, stream);

    luanch_cuda_kernel(
        where, stream, nx, ny, nxy, 0, mask, feat_ids, feat_values, conds, counts, output_indices);
    cudaStreamSynchronize(stream);

    *cpu_counts = *cpu_counts + 1;
    long long int* cpu_indices = nullptr;
    cudaHostAlloc(&cpu_indices, sizeof(long long int) * *cpu_counts * 2, cudaHostAllocDefault);
    assert(cpu_indices);
    cudaMemcpy(cpu_indices, output_indices, sizeof(long long int) * *cpu_counts * 2, cudaMemcpyDeviceToHost);

    std::vector<uint32_t> indices(*cpu_counts, 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [cpu_indices](uint32_t const lhs, uint32_t const rhs) {
        if (cpu_indices[2 * lhs] < cpu_indices[2 * rhs])
            return true;
        else if (cpu_indices[2 * lhs] == cpu_indices[2 * rhs])
            return cpu_indices[2 * lhs + 1] < cpu_indices[2 * rhs + 1];
        else
            return false;
    });

    std::vector<long long int> sorted_indices;
    for (auto const i : indices) {
        sorted_indices.push_back(cpu_indices[i * 2]);
        sorted_indices.push_back(cpu_indices[i * 2 + 1]);
    }

    cudaMemcpy(output_indices,
               sorted_indices.data(),
               sizeof(long long int) * *cpu_counts * 2,
               cudaMemcpyHostToDevice);

    luanch_cuda_kernel(select_feat,
                       stream,
                       nx,
                       (unsigned int)*cpu_counts,
                       (unsigned int)*cpu_counts,
                       0,
                       *cpu_counts,
                       conds,
                       output_indices,
                       feat_ids,
                       feat_values,
                       output_feat_ids,
                       output_feat_values);
    cudaStreamSynchronize(stream); // wait for done

    cudaFree(cpu_indices);
    cudaFree(cpu_counts);
}
