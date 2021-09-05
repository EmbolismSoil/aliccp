#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include <boost/algorithm/string.hpp>
#include <gflags/gflags.h>
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
    table_opt.block_cache = rocksdb::NewLRUCache(1000 * (1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));
    return rocksdb::DB::Open(opt, path, db);
}


static void
print_example(aliccp::Example const* example)
{
    fprintf(stderr,
            "Get example: example_id = %u, y = %d, z = %d, comm_feat_idx = %s, feat_num = %u, nfeats "
            "= %u\n",
            example->example_id(),
            example->y(),
            example->z(),
            example->comm_feat_id()->c_str(),
            example->feat_num(),
            example->feats()->Length());

    for (auto const& feat : *example->feats()) {
        fprintf(stderr,
                "Get feature: feat_field_id = %u, feat_id = %u, value = %f\n",
                feat->feat_field_id(),
                feat->feat_id(),
                feat->value());
    }
}

static void
print_comm_feats(aliccp::CommFeature const* comm_feats)
{
    fprintf(stderr,
            "Get example: comm_feat_idx = %s, feat_num = %u, nfeats "
            "= %u\n",
            comm_feats->comm_feat_id()->c_str(),
            comm_feats->feat_num(),
            comm_feats->feats()->Length());

    for (auto const& feat : *comm_feats->feats()) {
        fprintf(stderr,
                "Get feature: feat_field_id = %u, feat_id = %u, value = %f\n",
                feat->feat_field_id(),
                feat->feat_id(),
                feat->value());
    }
}

DEFINE_string(type, "", "[example|comm_feat]");
DEFINE_string(db, "", "Path to db");
DEFINE_string(keys, "", "key1,key2,...,keyn");

int
main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_type.empty() || FLAGS_db.empty() || FLAGS_keys.empty()) {
        fprintf(stderr, "type, db, keys are required\n");
        return -1;
    }

    rocksdb::DB* p;
    auto status = open_db(FLAGS_db.c_str(), &p);
    if (!status.ok()) {
        fprintf(stderr, "open db %s failed. what: %s\n", argv[1], status.ToString().c_str());
        return -1;
    }
    auto db = std::shared_ptr<rocksdb::DB>(p);
    std::vector<std::string> keys;
    boost::split(keys, FLAGS_keys, boost::is_any_of(","));

    auto const isexample = FLAGS_type == "example";

    for (auto const& key : keys) {
        std::string value;
        uint32_t u32key = 0;
        rocksdb::Slice skey;
        if (isexample) {
            u32key = static_cast<uint32_t>(std::stoul(key));
            skey = rocksdb::Slice(reinterpret_cast<char*>(&u32key), sizeof(u32key));
        } else {
            skey = key;
        }

        fprintf(stderr, "read key: 0x%s\n", skey.ToString(isexample).c_str());
        auto status = db->Get(rocksdb::ReadOptions(), skey, &value);
        if (!status.ok()) {
            fprintf(stderr,
                    "read from %s key = %s failed. message: %s\n",
                    key.c_str(),
                    skey.ToString(isexample).c_str(),
                    status.ToString().c_str());
            continue;
        }

        if (isexample) {
            auto example = aliccp::GetExample(value.data());
            print_example(example);
        } else {
            auto comm_feat = aliccp::GetCommFeature(value.data());
            print_comm_feats(comm_feat);
        }
    }
}
