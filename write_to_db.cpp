#include "comm_feats_generated.h"
#include "example_generated.h"
#include "feature_generated.h"
#include "vocab_generated.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/table.h>

static uint32_t
field_to_uint32(std::string const& field_id)
{
    if (field_id.find('_') != std::string::npos) {
        std::string buf;
        std::copy_if(field_id.cbegin(), field_id.cend(), std::back_inserter(buf), [](const char c) {
            return c != '_';
        });
        return std::stoul(buf);
    } else {
        return 100 * std::stoul(field_id);
    }
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
            "Get comm feats: comm_feat_idx = %s, feat_num = %u, nfeats "
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

static std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> field_stat;
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

        auto const feat_field_id = field_to_uint32(ids[0]);
        auto const feat_id = static_cast<uint32_t>(std::stoul(ids[1]));
        auto const value = std::stof(kv[1]);
        field_stat[feat_field_id][feat_id] += 1;
        vfeats.push_back(aliccp::CreateFeature(builder, feat_field_id, feat_id, value));
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
    opt.compression = rocksdb::kZlibCompression;

    rocksdb::BlockBasedTableOptions table_opt;
    table_opt.block_cache = rocksdb::NewLRUCache(1000 * (1024 * 1024));
    table_opt.block_cache_compressed = rocksdb::NewLRUCache(500 * (1024 * 1024));
    opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_opt));
    return rocksdb::DB::Open(opt, path, db);
}

void
dump_stat_info(std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> const& stat,
               std::string const& path)
{

    flatbuffers::FlatBufferBuilder builder(0);
    std::vector<aliccp::VocabEntry> entries;
    std::vector<aliccp::FieldInfo> infos;
    for (auto const& field : stat) {
        std::vector<std::pair<uint32_t, uint32_t>> feats;
        std::copy(field.second.cbegin(), field.second.cend(), std::back_inserter(feats));
        std::sort(feats.begin(),
                  feats.end(),
                  [](std::pair<uint32_t, uint32_t> const& lhs, std::pair<uint32_t, uint32_t> const& rhs) {
                      return lhs.second > rhs.second;
                  });

        uint32_t field_counts = 0;
        for (auto i = 0; i < feats.size(); ++i) {
            auto field_id = field.first;
            auto feat_id = feats[i].first;
            auto counts = feats[i].second;
            auto vocab_id = i + 1;
            entries.emplace_back(field_id, feat_id, vocab_id, counts);
            field_counts += counts;
        }
        
        infos.emplace_back(field.first, field.second.size(), field_counts);
    }
    auto vocab = aliccp::CreateVocabDirect(builder, &entries, &infos);
    builder.Finish(vocab);

    auto buf = builder.GetBufferPointer();
    auto size = builder.GetSize();
    std::ofstream ofile(path, std::ios::binary);
    ofile.write((char*)buf, size);
    ofile.close();
}

static int
write_features_to_db(const std::string& path_to_data,
                     const std::string& path_to_db,
                     const int batch_size,
                     bool const isexample)
{
    rocksdb::DB* db = nullptr;
    auto status = open_db(path_to_db.c_str(), &db);
    if (!status.ok()) {
        fprintf(stderr, "open db failed: %s, msg: %s\n", path_to_db.c_str(), status.ToString().c_str());
        return -1;
    }

    auto pdb = std::shared_ptr<rocksdb::DB>(db);

    rocksdb::WriteOptions option;

    std::ifstream ifs(path_to_data);
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

        total_size += (keybuf.size() + value.size());
        rocksdb::Slice key(keybuf.data(), keybuf.size());
        batch->Put(key, value);
        ++cnt;
        builder.Clear();

        if (cnt % batch_size == 0) {
            pdb->Write(option, &(*batch));
            fprintf(stderr,
                    "write %s db  batch size = %d, cnt = %d, total_writen_size = %lu, cost %ld seconds\n",
                    path_to_db.c_str(),
                    batch_size,
                    cnt,
                    total_size,
                    time(nullptr) - start);
            start = time(nullptr);
            batch = std::make_shared<rocksdb::WriteBatch>();
        }
    }
    if (batch->GetDataSize() > 0) pdb->Write(option, &(*batch));
    return 0;
}

DEFINE_string(common_data, "", "path to common feats data");
DEFINE_string(examples_data, "", "Path to examples data");
DEFINE_string(common_db, "", "Path to common feats db");
DEFINE_string(examples_db, "", "Path to examples db");
DEFINE_int32(batch, 10000, "batch size");
DEFINE_string(stat, "", "path to stat flatbuffers binary");

int
main(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_common_db.empty() || FLAGS_examples_db.empty() || FLAGS_common_data.empty() ||
        FLAGS_examples_data.empty() || FLAGS_stat.empty()) {
        fprintf(stderr, "type, data, db are required.\n");
        return -1;
    }

    write_features_to_db(FLAGS_common_data, FLAGS_common_db, FLAGS_batch, false);
    write_features_to_db(FLAGS_examples_data, FLAGS_examples_db, FLAGS_batch, true);
    dump_stat_info(field_stat, FLAGS_stat);
}
