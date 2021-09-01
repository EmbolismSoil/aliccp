GENERATEDS := feature_generated.h comm_feats_generated.h example_generated.h
FBS_IDL := feature.fbs comm_feats.fbs example.fbs
TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS= $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
ROCKSDB_LDFALGS := -L/home/seminelee/github/rocksdb -l:librocksdb.a  -lpthread
GFLAGS_LDFLAGS := -L/home/seminelee/github/gflags/build/lib/ -l:libgflags.a
CXX=/usr/bin/c++
INCLUDES=/home/seminelee/github/rocksdb/include/

all: read_from_db write_to_db aliccp_rocksdb_op.so
$(GENERATEDS) : $(FBS_IDL)
	flatc -c -b $(FBS_IDL)
	flatc --python -c -b $(FBS_IDL)

read_from_db: read_from_db.cpp $(GENERATEDS)
	$(CXX) -std=c++11 -g -O0 read_from_db.cpp -D_GLIBCXX_USE_CXX11_ABI=0 -I$(INCLUDES) $(GFLAGS_LDFLAGS) $(ROCKSDB_LDFALGS) -o $@ 

write_to_db: write_to_db.cpp $(GENERATEDS)
	$(CXX) -std=c++11 -g -O0 write_to_db.cpp -D_GLIBCXX_USE_CXX11_ABI=0 -I$(INCLUDES) $(GFLAGS_LDFLAGS) $(ROCKSDB_LDFALGS) -o $@

%.so: %.cpp
	$(CXX) -std=c++11 -shared $< -o $@ -fPIC -I$(INCLUDES) $(TF_CFLAGS) $(TF_LFLAGS) $(ROCKSDB_LDFALGS)


.PHONY: clean

clean:
	-rm $(GENERATEDS)
	-rm read_from_db
	-rm write_to_db
	-rm -rf aliccp
	-rm aliccp_rocksdb_op.so
