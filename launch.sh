#!/bin/bash

SRC_DIR=/ccc/dsku/nfs-server/user/cont001/ocre/roussela/work/Projects/bench/uvm-benchmarks-suite
RESULT_DIR=$(SRC_DIR)/results/$SLURM_JOB_PARTITION

cd $(SRC_DIR)/00-numa_memcpy-explicit
make clean all

mkdir -p $(RESULT_DIR)
cd $(RESULT_DIR)

for i in 1 5 10 50 100 250 500 1000 2500 5000
do
  TEST_DIR=$(RESULT_DIR)"/""$i""MB"
  mkdir -p $(TEST_DIR)
  mkdir -p $(TEST_DIR)"/csv"
  mkdir -p $(TEST_DIR)"/pdf"
  cd $(TEST_DIR)"/csv"

  $(SRC_DIR)/00-numa_memcpy-explicit/numa_explicit.exe $i

  cd $(SRC_DIR)/results
  python3 plot_all.py $SLURM_JOB_PARTITION $i
done
