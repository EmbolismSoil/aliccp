
_TOP_ := $(shell pwd)

TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS= $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

ROCKSDB_PATH := $(_TOP_)/thirdparty/rocksdb/
GFLAGS_PATH := $(_TOP_)/thirdparty/gflags/build/
FLATBUFFERS_PATH := $(_TOP_)/thirdparty/flatbuffers/build/

ROCKSDB_LDFALGS := -L$(ROCKSDB_PATH)/build -l:librocksdb.a  -lpthread
GFLAGS_LDFLAGS := -L$(GFLAGS_PATH)/lib -l:libgflags.a

LIB_ROCKSDB := $(ROCKSDB_PATH)/build/librocksdb.a
LIB_ROCKSDB_MAKEFILE := $(ROCKSDB_PATH)/build/Makefile
LIB_GFLAGS := $(GFLAGS_PATH)/lib/libgflags.a
LIB_GFLAGS_MAKEFILE := $(GFLAGS_PATH)/Makefile
FLATBUFFERS_MAKEFILE := $(FLATBUFFERS_PATH)/Makefile
ROCKSDB_COMPILE_OPT := -DWITH_BZ2=OFF
ROCKSDB_COMPILE_OPT += -DWITH_LZ4=OFF
ROCKSDB_COMPILE_OPT += -DWITH_SNAPPY=OFF
ROCKSDB_COMPILE_OPT += -DWITH_ZLIB=OFF
ROCKSDB_COMPILE_OPT += -DWITH_GFLAGS=OFF
ROCKSDB_COMPILE_OPT += -DCMAKE_CXX_FLAGS='-fPIC -D_GLIBCXX_USE_CXX11_ABI=0'
ROCKSDB_COMPILE_OPT += -DCMAKE_C_FLAGS='-fPIC'
ROCKSDB_COMPILE_OPT += -DCMAKE_BUILD_TYPE=Release

GFLAGS_COMPILE_OPT += -DCMAKE_CXX_FLAGS='-fPIC -D_GLIBCXX_USE_CXX11_ABI=0'
GFLAGS_COMPILE_OPT += -DCMAKE_C_FLAGS='-fPIC'

CXX=/usr/bin/c++
CMAKE ?= cmake
MAKE  ?= make
FLATC := $(FLATBUFFERS_PATH)/flatc

JOBS := -j10
GENERATEDS := feature_generated.h comm_feats_generated.h example_generated.h
FBS_IDL := feature.fbs comm_feats.fbs example.fbs

SHARD_LIB_FLAGS += -fPIC
CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
CXXFLAGS += -std=c++11

INCLUDES := -I$(ROCKSDB_PATH)/include/
INCLUDES += -I$(GFLAGS_PATH)/build/include/

all: read_from_db write_to_db aliccp_rocksdb_op.so
$(GENERATEDS) : $(FBS_IDL) $(FLATC)
	$(FLATC) -c -b $(FBS_IDL)

read_from_db: read_from_db.cpp $(GENERATEDS) $(LIB_ROCKSDB) $(LIB_GFLAGS)
	$(CXX) read_from_db.cpp $(CXXFLAGS) $(INCLUDES) $(GFLAGS_LDFLAGS) $(ROCKSDB_LDFALGS) -o $@ 

write_to_db: write_to_db.cpp $(GENERATEDS) $(LIB_ROCKSDB) $(LIB_GFLAGS)
	$(CXX)  write_to_db.cpp $(CXXFLAGS) $(INCLUDES) $(GFLAGS_LDFLAGS) $(ROCKSDB_LDFALGS) -o $@

%.so: %.cpp $(LIB_ROCKSDB)
	$(CXX) -shared $< -o $@ $(INCLUDES) $(SHARD_LIB_FLAGS) $(CXXFLAGS) $(TF_CFLAGS) $(TF_LFLAGS) $(ROCKSDB_LDFALGS)

$(LIB_ROCKSDB): $(LIB_ROCKSDB_MAKEFILE)
	$(MAKE) -C $(ROCKSDB_PATH)/build $(JOBS)

$(LIB_GFLAGS): $(LIB_GFLAGS_MAKEFILE)
	$(MAKE) -C $(GFLAGS_PATH) $(JOBS)

$(FLATC): $(FLATBUFFERS_MAKEFILE)
	$(MAKE) -C $(FLATBUFFERS_PATH) $(JOBS)

$(LIB_ROCKSDB_MAKEFILE):
	- mkdir $(ROCKSDB_PATH)/build
	- cd $(ROCKSDB_PATH)/build && $(CMAKE) .. $(ROCKSDB_COMPILE_OPT)

$(FLATBUFFERS_MAKEFILE):
	- mkdir $(FLATBUFFERS_PATH)
	- cd $(FLATBUFFERS_PATH) && $(CMAKE) ..

$(LIB_GFLAGS_MAKEFILE):
	- mkdir $(GFLAGS_PATH)
	- cd $(GFLAGS_PATH) && $(CMAKE) .. $(GFLAGS_COMPILE_OPT)

.PHONY: clean

clean:
	-rm $(GENERATEDS)
	-rm read_from_db
	-rm write_to_db
	-rm aliccp_rocksdb_op.so
	-$(MAKE) -C $(ROCKSDB_PATH)/build clean
	-$(MAKE) -C $(GFLAGS_PATH) clean
	-$(MAKE) -C $(FLATBUFFERS_PATH) clean
