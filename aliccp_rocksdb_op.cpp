#include "Timer.h"
#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <functional>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>
#include <unordered_set>

static rocksdb::Status
open_db(const char* path, rocksdb::DB** db)
{
    rocksdb::Options opt;
    opt.create_if_missing = false;
    opt.max_open_files = 3000;
    opt.max_write_buffer_number = 3;
    opt.target_file_size_base = 67108864;
    opt.new_table_reader_for_compaction_inputs = true;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache((1024 * 1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));

    return rocksdb::DB::OpenForReadOnly(opt, path, db);
}

namespace std {
template<>
struct hash<rocksdb::Slice>
{
    size_t operator()(rocksdb::Slice const& s) const noexcept { return hash_op(s.ToString(true)); }

  private:
    std::hash<std::string> hash_op;
};
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

        OP_REQUIRES_OK(context, context->GetAttr("max_feats", &max_feats_));

        rocksdb::DB* db;
        auto status = open_db(examples_db.c_str(), &db);
        if (!status.ok()) {
            context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
        }
        example_db_ = std::shared_ptr<rocksdb::DB>(db);

        status = open_db(comm_feats_db.c_str(), &db);
        if (!status.ok()) {
            context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
        }
        comm_feats_db_ = std::shared_ptr<rocksdb::DB>(db);

        read_opts_.readahead_size = 1024 * 1024 * 1024;
    }

    static Tensor* alloc_tensor(OpKernelContext* context, std::vector<int32> const& dims, int const i)
    {
        TensorShape shape;
        auto status = TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
        if (!status.ok()) {
            context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
            return nullptr;
        }

        Tensor* tensor = nullptr;
        status = context->allocate_output(i, shape, &tensor);
        if (!status.ok()) {
            context->CtxFailure(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT, status.ToString()));
            return nullptr;
        }

        return tensor;
    }

    void parse_examples(OpKernelContext* context,
                        std::vector<const aliccp::Example*> const& examples,
                        std::unordered_map<std::string, const aliccp::CommFeature*> const& comm_feats,
                        Tensor* field_id_tensor,
                        Tensor* feat_id_tensor,
                        Tensor* feats_tensor,
                        Tensor* y,
                        Tensor* z,
                        Tensor* lens_tensor)
    {
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

        auto batch_size = 64;
        auto batch_nums = (examples.size() + batch_size - 1) / batch_size;

        auto parse_batch = [this,
                            batch_size,
                            examples,
                            comm_feats,
                            &field_id_matrix,
                            &feat_matrix,
                            &feat_id_matrix,
                            &y,
                            &z,
                            &lens_tensor,
                            field_to_int64](Eigen::Index start, Eigen::Index end) {
            start = std::min(start * batch_size, (Eigen::Index)examples.size());
            end = std::min(end * batch_size, (Eigen::Index)examples.size());

            for (auto i = start; i < end; ++i) {
                auto const example = examples[i];
                if (!example) {
                    continue;
                }

                y->flat<int64>()(i) = static_cast<int64>(example->y());
                z->flat<int64>()(i) = static_cast<int64>(example->z());

                auto feats = example->feats();

                int k = 0;
                int len = std::min((int32)feats->Length(), max_feats_);
                for (; k < len; ++k) {
                    auto feat = feats->Get(k);
                    auto field_id = feat->feat_field_id();
                    auto feat_id = feat->feat_id();
                    auto value = feat->value();

                    feat_matrix(i, k) = value;
                    field_id_matrix(i, k) = field_to_int64(field_id->str());
                    feat_id_matrix(i, k) = feat_id;
                }

                auto comm_feat_id = example->comm_feat_id();
                auto iter = comm_feats.find(comm_feat_id->str());

                if (iter != comm_feats.cend()) {
                    auto const comm_feat = iter->second->feats();
                    auto cap = std::min(max_feats_, (int32)(comm_feat->Length() + feats->Length()));
                    for (; k < cap; ++k) {
                        auto feat = comm_feat->Get(k - len);
                        auto field_id = feat->feat_field_id();
                        auto feat_id = feat->feat_id();
                        auto value = feat->value();

                        feat_matrix(i, k) = value;
                        field_id_matrix(i, k) = field_to_int64(field_id->str());
                        feat_id_matrix(i, k) = feat_id;
                    }
                }

                lens_tensor->flat<int64>()(i) = k;
                for (; k < max_feats_; ++k) {
                    feat_matrix(i, k) = 0.0;
                    field_id_matrix(i, k) = 0;
                    feat_id_matrix(i, k) = 0;
                }
            }
        };

        auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;
        auto cost_per_unit = 10 * 6000 * batch_size;
        thread_pool->ParallelFor(batch_nums, cost_per_unit, std::move(parse_batch));
    }

    void Compute(OpKernelContext* context) override
    {
        auto const input = context->input(0);

        if (input.dims() != 1) {
            context->CtxFailure(
                __FILE__, __LINE__, Status(error::INVALID_ARGUMENT, "1d tensor is accepted only."));
            return;
        }

        std::vector<std::string> buf;
        OP_REQUIRES_OK(context, read_values(input, buf));

        auto const nelems = static_cast<int32>(input.NumElements());

        auto field_id_tensor = alloc_tensor(context, { nelems, max_feats_ }, 0);
        auto feat_id_tensor = alloc_tensor(context, { nelems, max_feats_ }, 1);
        auto feats_tensor = alloc_tensor(context, { nelems, max_feats_ }, 2);
        auto y = alloc_tensor(context, { nelems }, 3);
        auto z = alloc_tensor(context, { nelems }, 4);
        auto lens_tensor = alloc_tensor(context, { nelems }, 5);

        if (!field_id_tensor || !feat_id_tensor || !feats_tensor || !y || !z || !lens_tensor) {
            return;
        }

        std::vector<const aliccp::Example*> examples;
        std::transform(buf.cbegin(), buf.cend(), std::back_inserter(examples), [](std::string const& s) {
            return aliccp::GetExample(s.data());
        });

        std::unordered_set<rocksdb::Slice> comm_keys;
        std::transform(examples.cbegin(),
                       examples.cend(),
                       std::inserter(comm_keys, comm_keys.end()),
                       [](const aliccp::Example* example) {
                           auto key = rocksdb::Slice(example->comm_feat_id()->c_str(),
                                                     example->comm_feat_id()->Length());
                           return key;
                       });

        std::vector<string> comm_feats_buf;
        std::unordered_map<std::string, const aliccp::CommFeature*> comm_feats;
        OP_REQUIRES_OK(context,
                       read_db(comm_feats_db_,
                               std::vector<rocksdb::Slice>(comm_keys.begin(), comm_keys.end()),
                               comm_feats_buf));

        std::transform(comm_feats_buf.cbegin(),
                       comm_feats_buf.cend(),
                       std::inserter(comm_feats, comm_feats.end()),
                       [](std::string const& s) {
                           auto comm_feat = aliccp::GetCommFeature(s.data());
                           return std::make_pair(comm_feat->comm_feat_id()->str(), comm_feat);
                       });

        Timer timer;
        parse_examples(
            context, examples, comm_feats, field_id_tensor, feat_id_tensor, feats_tensor, y, z, lens_tensor);
    }

  private:
    Status read_db(std::shared_ptr<rocksdb::DB> db,
                   std::vector<rocksdb::Slice> const& keys,
                   std::vector<std::string>& values)
    {
        Timer timer;
        auto status = db->MultiGet(read_opts_, keys, &values);

        int failed = 0;
        for (auto i = 0; i < keys.size(); ++i) {
            auto s = status[i];
            if (!s.ok()) {
                std::string msg = s.ToString() + ": key = " + keys[i].ToString(true);
                return Status(error::DATA_LOSS, msg);
            }
        }

        return Status::OK();
    }

    Status read_values(Tensor const& input, std::vector<std::string>& values)
    {
        auto nelems = input.NumElements();
        auto input_flat = input.flat<int64>();
        std::vector<rocksdb::Slice> keys;

        uint32_t* keybuf = (uint32_t*)::malloc(sizeof(uint32_t) * nelems);
        if (!keybuf) {
            return Status(error::INTERNAL, "malloc buf failed.");
        }

        for (auto i = 0; i < nelems; ++i) {
            auto example_id = static_cast<uint32_t>(input_flat(i));
            keybuf[i] = example_id;
            const char* p = reinterpret_cast<const char*>(keybuf + i);
            rocksdb::Slice key(p, sizeof(example_id));
            keys.push_back(key);
        }

        auto status = read_db(example_db_, keys, values);
        free(keybuf);
        return status;
    }

    std::shared_ptr<rocksdb::DB> example_db_;
    std::shared_ptr<rocksdb::DB> comm_feats_db_;
    rocksdb::ReadOptions read_opts_;
    int32 max_feats_;
};
REGISTER_KERNEL_BUILDER(Name("AliCCPRocksDB").Device(DEVICE_CPU), AliCCPRocksDBOp);
};

