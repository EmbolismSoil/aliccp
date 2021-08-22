#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("AliCCPRocksDB").Input("example_ids: int32").Output("features: float32");

class AliCCPRocksDBOp : OpKernel
{
  public:
    explicit AliCCPRocksDBOp(OpKernelConstruction* context)
        : OpKernel(context)
    {}

    void Compute(OpKernelContext* context) override {}
};
};
