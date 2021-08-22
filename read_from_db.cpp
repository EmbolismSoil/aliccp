#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>

static rocksdb::Status
open_db(const char* path, rocksdb::DB** db)
{
    rocksdb::Options opt;
    rocksdb::BlockBasedTableOptions table_opt;
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
                "Get feature: feat_field_id = %s, feat_id = %u, value = %f\n",
                feat->feat_field_id()->c_str(),
                feat->feat_id(),
                feat->value());
    }
}

int
main(int argc, const char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage %s <path to db> key1[,key2...]\n", argv[0]);
        return -1;
    }

    rocksdb::DB* db;
    auto status = open_db(argv[1], &db);
    if(!status.ok()){
        fprintf(stderr, "open db %s failed. what: %s\n", argv[1], status.ToString().c_str());
        return -1;
    }

    for (auto i = 2; i < argc; ++i) {
        std::string value;
        auto example_id = static_cast<uint32_t>(std::stoul(argv[i]));
        auto key = rocksdb::Slice(reinterpret_cast<char*>(&example_id), sizeof(example_id));
        fprintf(stderr, "read key: 0x%s\n", key.ToString(true).c_str());
        auto status = db->Get(
            rocksdb::ReadOptions(), key, &value);
        if (!status.ok()) {
            fprintf(stderr, "read from %s key = %u failed.\n", argv[1], example_id);
            continue;
        }

        auto example = aliccp::GetExample(value.data());
        print_example(example);
    }
}
