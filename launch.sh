#!/bin/bash

SRC_DIR="/ccc/dsku/nfs-server/user/cont001/ocre/roussela/work/Projects/bench/uvm-benchmarks-suite"
RESULT_DIR="$SRC_DIR/results/$SLURM_JOB_PARTITION"

cd $SRC_DIR/00-numa_memcpy-explicit
make clean all

cd $SRC_DIR/01-numa_memcpy-managed
make clean all

cd $SRC_DIR/02-numa_memcpy-implicit-mimic
make clean all

cd $SRC_DIR/03-numa_memcpyasync/
make clean all

mkdir -p $RESULT_DIR
cd $RESULT_DIR

# for i in 1 5 10 50 100 250 500 1000 2500 5000
for i in 1 5 10 50 100
do
  TEST_DIR="$RESULT_DIR/$i"
  TEST_DIR+="MB"
  mkdir -p "$TEST_DIR"
  mkdir -p "$TEST_DIR/csv"
  mkdir -p "$TEST_DIR/pdf"
  cd "$TEST_DIR/csv"

  $SRC_DIR/00-numa_memcpy-explicit/numa_explicit.exe -s $i
  $SRC_DIR/01-numa_memcpy-managed/numa_implicit.exe -s $i
  $SRC_DIR/02-numa_memcpy-implicit-mimic/numa_implicit-mimic.exe -s $i
  $SRC_DIR/03-numa_memcpyasync/numa_memcpy-async.exe -s $i

  res_dir="$SRC_DIR/results"
  cd $res_dir
  python3 plot_all.py $SLURM_JOB_PARTITION $i
done
