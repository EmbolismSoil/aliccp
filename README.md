# aliccp

AliCCP数据集通过flatsbuffer格式存入rocksdb中，训练时通过特定op构造样本，从而加快整体训练速度

## 编译
1. git clone --recursive https://git.conleylee.com/conley/aliccp.git
2. make

## `write_to_db`
此工具用于将数据写入db, 对于`sample_skeleton_train.csv`使用exampleid作为key，而`common_features_train.csv`使用`comm_feat_id`作为key, 命令行包含如下参数
```bash
    -batch (单次刷入磁盘的batch大小) type: int32 default: 10000
    -common_data (数据集common_features_train.csv的路径) type: string default: ""
    -common_db (数据集common_features_train.csv写入磁盘的数据库) type: string default: ""
    -examples_data (数据集sample_skeleton_train.csv的路径) type: string default: ""
    -examples_db (sample_skeleton_train.csv数据集写入磁盘的数据库) type: string default: ""
    -stat (vocab统计文件的路径) type: string default: ""
```
使用方式
```bash
./write_to_db -batch 10000 -common_data ../common_features_train.csv -common_db ../common_feats.db -examples_data ../sample_skeleton_train.csv -examples_db ../examples.db -stat ./field_feat_vocab.bin
```
其中vocab需要传给op，以便将`feat_id`转换成`[1, slots]`范围内的index，从而能在tensorflow中做lookup操作。vocab中存放的`slots`记录词表大小，用于设置embedding矩阵的size

## `read_from_db`
此工具可以通过命令行将数据从db中读出,使用方式
```bash
read_from_db -db <path to db> -type [example|comm_feat] -keys 'key1,key2,...,keyn'
```
例如使用
```bash
./read_from_db -db examples.db -type 'example' -keys '1,2,3,4,5'
```
从examples.db中读取key=1,2,3,4,5的5个example

## `aliccp_rocksdb_op.so`
此动态库是tensorflow op用于训练时从db中读取训练数据，输入为`example_id`,`examples_db`,`comm_feats_db`,`max_feats`，其中`max_feats`是pad长度，如果一个样本的总特征个数小于`max_feats`则会用0补长到此长度，如果大于此长度则进行截断。`lens`是补长前的特征长度。输出为拼接上`comm_feats`的训练样本，输出格式为`feature_field_id, feature_id, feature_values, y, z, lens`,下面是一个读取exampleid为1至50000训练样本的例子
```python
import tensorflow as tf
ops = tf.load_op_library('./aliccp_rocksdb_op.so')
ops.ali_ccp_rocks_db(range(1, 50000), examples_db='examples.db', comm_feats_db='common_feats.db', max_feats=1000, vocab='field_feat_vocab.bin')
```
使用dataset按照batch=1024读取50w训练样本:
```python
def example_ids():
    for i in range(1, 500000):
        yield i
ds = tf.data.Dataset.from_generator(example_ids, output_types=tf.int64).prefetch(20000).batch(1024)
examples = ds.map(lambda x: ops.ali_ccp_rocks_db(x, examples_db='examples.db', comm_feats_db='common_feats.db', max_feats=1000, vocab='field_feat_vocab.bin'), num_parallel_calls=16)
```

## 体积
整个`common_features_train.csv`存放到rocksdb中占用3.3G磁盘大小，`sample_skeleton_train.csv`存放到rocksdb中占用5.8G大小, 如果使用tfrecord来存放训练样本，则需要500G大小，相比之下rocksdb压缩储存体积减小50倍有余

## 性能
相比tfrecord底层protobuf的储存，体积小得多，不需要大量磁盘io，而且flatbuffers反序列化相比protobuf来说十分轻量，总而言之就是快！具体测试数据待补充
