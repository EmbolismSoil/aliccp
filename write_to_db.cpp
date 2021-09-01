#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>

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

static void
print_comm_feats(aliccp::CommFeature const* comm_feats)
{
    fprintf(stderr,
            "Get comm feats: comm_feat_idx = %s, feat_num = %u, nfeats "
            "= %u\n",
            comm_feats->comm_feat_id()->c_str(),
            comm_feats->feat_num(),
            comm_feats->feats()->Length());

    for (auto const& feat : *comm_feats->feats()) {
        fprintf(stderr,
                "Get feature: feat_field_id = %s, feat_id = %u, value = %f\n",
                feat->feat_field_id()->c_str(),
                feat->feat_id(),
                feat->value());
    }
}

static int
parse_feats(flatbuffers::FlatBufferBuilder& builder,
            std::string const& line,
            std::vector<flatbuffers::Offset<aliccp::Feature>>& vfeats)
{
    std::vector<std::string> feats;
    boost::split(feats, line, boost::is_any_of("\x01"));
    if (feats.empty()) {
        return -1;
    }

    for (auto const& feat : feats) {
        std::vector<std::string> kv;
        boost::split(kv, feat, boost::is_any_of("\x03"));
        if (kv.size() != 2) {
            fprintf(stderr, "kv.size(=%zu) != 2, feat = %s\n", kv.size(), feat.c_str());
            continue;
        }

        std::vector<std::string> ids;
        boost::split(ids, kv[0], boost::is_any_of("\x02"));
        if (ids.size() != 2) {
            fprintf(stderr, "ids.size(=%zu) != 2, kv[0] = %s", ids.size(), kv[0].c_str());
            continue;
        }

        auto const& feat_field_id = ids[0];
        auto const feat_id = static_cast<uint32_t>(std::stoul(ids[1]));
        auto const value = std::stof(kv[1]);
        vfeats.push_back(aliccp::CreateFeatureDirect(builder, feat_field_id.c_str(), feat_id, value));
    }

    return 0;
}

static int
parse_skeleton_line(flatbuffers::FlatBufferBuilder& builder, std::string const& line, std::vector<char>& key)
{
    std::vector<std::string> items;
    boost::split(items, line, boost::is_any_of(","));
    if (items.size() != 6) {
        fprintf(stderr, "items.size(=%zu) != 6\n", items.size());
        return -1;
    }

    auto const example_id = static_cast<uint32_t>(std::stoul(items[0]));
    auto const y = static_cast<uint16_t>(std::stoi(items[1]));
    auto const z = static_cast<uint16_t>(std::stoi(items[2]));
    auto const& feat_idx = items[3];
    auto const feat_num = static_cast<uint16_t>(std::stoul(items[4]));
    auto const& feats = items[5];
    const char* p = reinterpret_cast<const char*>(&example_id);
    for (auto i = 0; i < sizeof(example_id); ++i) {
        key.push_back(p[i]);
    }

    std::vector<flatbuffers::Offset<aliccp::Feature>> vfeats;
    if (parse_feats(builder, feats, vfeats) != 0) {
        return -1;
    }

    auto example =
        aliccp::CreateExampleDirect(builder, example_id, y, z, feat_idx.c_str(), feat_num, &vfeats);
    builder.Finish(example);
    return 0;
}

static int
parse_common_line(flatbuffers::FlatBufferBuilder& builder, std::string const& line, std::vector<char>& key)
{
    std::vector<std::string> items;
    boost::split(items, line, boost::is_any_of(","));
    if (items.size() != 3) {
        fprintf(
            stderr, "parse_common_line failed. items.size(=%zu) != 3, line=%s\n", items.size(), line.c_str());
        return -1;
    }

    auto const& comm_feat_id = items[0];
    auto const feat_num = static_cast<uint16_t>(std::stoul(items[1]));
    auto const& feats = items[2];

    std::copy(comm_feat_id.cbegin(), comm_feat_id.cend(), std::back_inserter(key));
    std::vector<flatbuffers::Offset<aliccp::Feature>> vfeats;
    if (parse_feats(builder, feats, vfeats) != 0) {
        fprintf(stderr, "parse comm_feat feats failed. line = %s\n", feats.c_str());
        return -1;
    }

    auto comm_feats = aliccp::CreateCommFeatureDirect(builder, comm_feat_id.c_str(), feat_num, &vfeats);
    builder.Finish(comm_feats);
    return 0;
}

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

DEFINE_string(type, "", "[example|comm_feat]");
DEFINE_string(data, "", "path to data");
DEFINE_string(db, "", "Path to db");
DEFINE_int32(batch, 10000, "batch size");
DEFINE_int64(total, -1, "total records");

int
main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_type.empty() || FLAGS_data.empty() || FLAGS_db.empty()) {
        fprintf(stderr, "type, data, db are required.\n");
        return -1;
    }

    if (FLAGS_type != "example" && FLAGS_type != "comm_feat") {
        fprintf(stderr, "type should be example or comm_feat\n");
        return -1;
    }

    rocksdb::DB* db = nullptr;
    if (!open_db(FLAGS_db.c_str(), &db).ok()) {
        fprintf(stderr, "open db failed: %s\n", FLAGS_db.c_str());
        return -1;
    }

    auto pdb = std::shared_ptr<rocksdb::DB>(db);

    bool const isexample = FLAGS_type == "example";

    rocksdb::WriteOptions option;

    std::ifstream ifs(FLAGS_data);
    std::string line;
    int cnt = 0;
    flatbuffers::FlatBufferBuilder builder(0);

    auto batch = std::make_shared<rocksdb::WriteBatch>();
    auto start = time(nullptr);
    uint64_t total_size = 0;
    while (std::getline(ifs, line)) {
        std::vector<char> keybuf;
        if (isexample) {
            parse_skeleton_line(builder, line, keybuf);
        } else {
            parse_common_line(builder, line, keybuf);
        }

        auto buf = builder.GetBufferSpan();
        rocksdb::Slice value(reinterpret_cast<char*>(buf.data()), buf.size());
        if (isexample) {
            // print_example(aliccp::GetExample(buf.data()));
        } else {
            // print_comm_feats(aliccp::GetCommFeature(buf.data()));
        }
        
        total_size += (keybuf.size() + value.size());
        rocksdb::Slice key(keybuf.data(), keybuf.size());
        batch->Put(key, value);
        ++cnt;
        builder.Clear();

        if (cnt % FLAGS_batch == 0) {
            pdb->Write(option, &(*batch));
            fprintf(stderr,
                    "write batch size = %d, cnt = %d, total_writen_size = %lu, cost %ld seconds\n",
                    FLAGS_batch,
                    cnt,
                    total_size,
                    time(nullptr) - start);
            start = time(nullptr);
            batch = std::make_shared<rocksdb::WriteBatch>();
        }

        if ((FLAGS_total > 0) && cnt >= FLAGS_total){
            break;
        }
    }
    if (batch->GetDataSize() > 0) pdb->Write(option, &(*batch));
}
