# aliccp

AliCCP数据集通过flatsbuffer格式存入rocksdb中，训练时通过特定op构造样本，从而加快整体训练速度

## 编译
1. git clone --recursive https://git.conleylee.com/conley/aliccp.git
2. make

## `write_to_db`
此工具用于将数据写入db, 对于`sample_skeleton_train.csv`使用exampleid作为key，而`common_features_train.csv`使用`comm_feat_id`作为key  
使用方式
```bash
write_to_db -batch <batch size> -data <path to data file> -db <path to db> -type [example|comm_feat]
```
当`type=example`时，`-data`参数应当指向`sample_skeleton_train.csv`文件，当`type=comm_feat`时，则应当指向`common_features_train.csv`参数。

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
ops.ali_ccp_rocks_db(range(1, 50000), examples_db='examples.db', comm_feats_db='common_feats.db', max_feats=1000)
```
使用dataset按照batch=1024读取50w训练样本:
```python
def example_ids():
    for i in range(1, 500000):
        yield i
examples = tf.data.Dataset.from_generator(example_ids, output_types=tf.int64).batch(1024).map(lambda x: ops.ali_ccp_rocks_db(x, examples_db='../examples.db', comm_feats_
db='../common_feats.db', max_feats=1000))
```

## 性能
