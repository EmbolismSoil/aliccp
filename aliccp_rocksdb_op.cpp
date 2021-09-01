#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>

static rocksdb::Status
open_db(const char* path, rocksdb::DB** db)
{
    rocksdb::Options opt;
    opt.create_if_missing = true;
    opt.max_open_files = 3000;
    opt.write_buffer_size = 500 * 1024 * 1024;
    opt.max_write_buffer_number = 3;
    opt.target_file_size_base = 67108864;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache((1024 * 1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));

    return rocksdb::DB::Open(opt, path, db);
}

namespace tensorflow {
REGISTER_OP("AliCCPRocksDB")
    .Input("example_ids: int64")
    .Output("feat_field_id: int64")
    .Output("feat_id: int64")
    .Output("features: float32")
    .Output("y: int64")
    .Output("z: int64")
    .Output("lens: int64")
    .Attr("examples_db: string")
    .Attr("comm_feats_db: string")
    .Attr("max_feats: int");

class AliCCPRocksDBOp : public OpKernel
{
  public:
    explicit AliCCPRocksDBOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
        std::string examples_db;
        OP_REQUIRES_OK(context, context->GetAttr("examples_db", &examples_db));

        std::string comm_feats_db;
        OP_REQUIRES_OK(context, context->GetAttr("comm_feats_db", &comm_feats_db));

        OP_REQUIRES_OK(context, context->GetAttr("max_feats", &max_feats));

        rocksdb::DB* db;
        auto status = open_db(examples_db.c_str(), &db);
        if (!status.ok()) {
            context->CtxFailure(Status(error::INVALID_ARGUMENT, status.ToString()));
        }
        example_db_ = std::shared_ptr<rocksdb::DB>(db);

        status = open_db(comm_feats_db.c_str(), &db);
        if (!status.ok()) {
            context->CtxFailure(Status(error::INVALID_ARGUMENT, status.ToString()));
        }
        comm_feats_db_ = std::shared_ptr<rocksdb::DB>(db);
    }

    static Tensor* alloc_tensor(OpKernelContext* context, std::vector<int32> const& dims, int const i)
    {
        TensorShape shape;
        auto status = TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
        if (!status.ok()) {
            context->CtxFailure(Status(error::INVALID_ARGUMENT, status.ToString()));
            return nullptr;
        }

        Tensor* tensor = nullptr;
        status = context->allocate_output(i, shape, &tensor);
        if (!status.ok()) {
            context->CtxFailure(Status(error::INVALID_ARGUMENT, status.ToString()));
            return nullptr;
        }

        return tensor;
    }

    void Compute(OpKernelContext* context) override
    {
        auto const input = context->input(0);

        if (input.dims() != 1) {
            context->CtxFailure(Status(error::INVALID_ARGUMENT, "1d tensor is accepted only."));
            return;
        }

        std::vector<std::string> buf;
        read_values(input, buf);

        auto const nelems = static_cast<int32>(input.NumElements());

        auto field_id_tensor = alloc_tensor(context, { nelems, max_feats }, 0);
        auto feat_id_tensor = alloc_tensor(context, { nelems, max_feats }, 1);
        auto feats_tensor = alloc_tensor(context, { nelems, max_feats }, 2);
        auto y = alloc_tensor(context, { nelems }, 3);
        auto z = alloc_tensor(context, { nelems }, 4);
        auto lens_tensor = alloc_tensor(context, { nelems }, 5);

        if (!field_id_tensor || !feat_id_tensor || !feats_tensor || !y || !z || !lens_tensor) {
            return;
        }

        auto feat_matrix = feats_tensor->matrix<float>();
        auto field_id_matrix = field_id_tensor->matrix<int64>();
        auto feat_id_matrix = feat_id_tensor->matrix<int64>();

        auto field_to_int64 = [](std::string const& field_id) {
            if (field_id.find('_') != std::string::npos) {
                std::string buf;
                std::copy_if(field_id.cbegin(), field_id.cend(), std::back_inserter(buf), [](const char c) {
                    return c != '_';
                });
                return std::stol(buf);
            } else {
                return 100 * std::stol(field_id);
            }
        };

        for (auto i = 0; i < buf.size(); ++i) {
            auto const& data = buf[i];
            if (data.empty()) {
                continue;
            }

            auto example = aliccp::GetExample(data.data());

            y->flat<int64>()(i) = static_cast<int64>(example->y());
            z->flat<int64>()(i) = static_cast<int64>(example->z());

            auto feats = example->feats();
            lens_tensor->flat<int64>()(i) = std::min((int32)feats->Length(), max_feats);

            for (auto k = 0; k < max_feats; ++k) {
                if (k >= feats->Length()) {
                    feat_matrix(i, k) = 0.0;
                    field_id_matrix(i, k) = 0;
                    feat_id_matrix(i, k) = 0;
                    continue;
                }

                auto feat = feats->Get(k);
                auto field_id = feat->feat_field_id();
                auto feat_id = feat->feat_id();
                auto value = feat->value();

                feat_matrix(i, k) = value;
                field_id_matrix(i, k) = field_to_int64(field_id->str());
                feat_id_matrix(i, k) = feat_id;
            }
        }
    }

  private:
    void read_values(Tensor const& input, std::vector<std::string>& values)
    {
        auto nelems = input.NumElements();
        auto input_flat = input.flat<int64>();
        std::vector<rocksdb::Slice> keys;

        uint32_t* keybuf = (uint32_t*)::malloc(sizeof(uint32_t)*nelems);
        for (auto i = 0; i < nelems; ++i) {
            auto example_id = static_cast<uint32_t>(input_flat(i));
            keybuf[i] = example_id;
            const char* p = reinterpret_cast<const char*>(keybuf + i);

            rocksdb::Slice key(p, sizeof(example_id));
            LOG(INFO) << "enqueue key " << key.ToString(true);
            keys.push_back(key);
        }

        auto status = example_db_->MultiGet(rocksdb::ReadOptions(), keys, &values);
        for (auto i = 0; i < input.NumElements(); ++i) {
            auto s = status[i];
            LOG(INFO) << "read example_id = " << keybuf[i] << "(hex=" << keys[i].ToString(true) << ") "
                      << (s.ok() ? "succ" : "failed :") << (s.ok() ? "" : status[i].ToString());
        }

        free(keybuf);
    }

    std::shared_ptr<rocksdb::DB> example_db_;
    std::shared_ptr<rocksdb::DB> comm_feats_db_;
    int32 max_feats;
};
REGISTER_KERNEL_BUILDER(Name("AliCCPRocksDB").Device(DEVICE_CPU), AliCCPRocksDBOp);
};

