#ifndef __FIELD_SELECT_KERNEK_H__
#define __FIELD_SELECT_KERNEK_H__

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"
#include <cuda.h>
#include <functional>
#include <utility>

template<typename Device>
struct FieldSelectFuctor;

template<>
struct FieldSelectFuctor<Eigen::GpuDevice>
{
    FieldSelectFuctor(Eigen::GpuDevice const& device);
    void launch_target_field_mask(unsigned int const nx,
                                  unsigned int const ny,
                                  unsigned int const nxy,
                                  long long int const* target_field,
                                  long long int const* field_ids,
                                  int* counts,
                                  int* conds,
                                  int* mask,
                                  int& cpu_counts);

    void luanch_select_feat(unsigned int const nx,
                            unsigned int const ny,
                            unsigned int const nxy,
                            int const* mask,
                            long long int const* feat_ids,
                            float const* feat_values,
                            int const* conds,
                            int* counts,
                            long long int* output_indices,
                            long long int* output_feat_ids,
                            float* output_feat_values);

  private:
    Eigen::GpuDevice const& device_;
};

#endif
