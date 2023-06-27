#!/bin/bash

SCRIPT=$(readlink -f $0)
SRC_DIR=$(dirname  $SCRIPT)

if ! test -d $SRC_DIR ; then
  echo "Invalid Source directory given: $SRC_DIR"
  exit
fi

RESULT_DIR="$SRC_DIR/results/$SLURM_JOB_PARTITION"

cd $SRC_DIR/00-numa_memcpy-explicit
make clean all

cd $SRC_DIR/01-numa_memcpy-managed
make clean all

cd $SRC_DIR/02-numa_memcpy-implicit-mimic
make clean all

cd $SRC_DIR/03-numa_memcpyasync/
make clean all

cd $SRC_DIR/04-numa_hostRegister/
make clean all

mkdir -p $RESULT_DIR
cd $RESULT_DIR

# for i in 1 5 10 50 100 250 500 1000 2500 5000
for i in 5 10 50 100
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
  $SRC_DIR/04-numa_hostRegister/numa_hostregister.exe -s $i

  #res_dir="$SRC_DIR/results"
  #cd $res_dir
  #python3 plot_all.py $SLURM_JOB_PARTITION $i
done
