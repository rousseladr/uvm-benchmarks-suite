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

cd $SRC_DIR/03-numa_memcpyasync/
make clean all

mkdir -p $RESULT_DIR
cd $RESULT_DIR

NITER=25

for i in 1 5 10 25 50 100 250 500 600
do
  TEST_DIR="$RESULT_DIR/$i"
  TEST_DIR+="MB"
  mkdir -p "$TEST_DIR"
  mkdir -p "$TEST_DIR/csv"
  mkdir -p "$TEST_DIR/pdf"
  cd "$TEST_DIR/csv"

  $SRC_DIR/00-numa_memcpy-explicit/numa_explicit.exe -i $NITER -s $i
  $SRC_DIR/00-numa_memcpy-explicit/numa_explicit.exe -i $NITER -s $i -c -m
  $SRC_DIR/00-numa_memcpy-explicit/numa_explicit.exe -i $NITER -s $i -c -d
  $SRC_DIR/00-numa_memcpy-explicit/numa_explicit.exe -i $NITER -s $i -c -d -m

  $SRC_DIR/03-numa_memcpyasync/numa_memcpy-async.exe -i $NITER -s $i
  $SRC_DIR/03-numa_memcpyasync/numa_memcpy-async.exe -i $NITER -s $i -c -m
  $SRC_DIR/03-numa_memcpyasync/numa_memcpy-async.exe -i $NITER -s $i -c -d
  $SRC_DIR/03-numa_memcpyasync/numa_memcpy-async.exe -i $NITER -s $i -c -d -m

  res_dir="$SRC_DIR/results"
  cd $res_dir
  python3 plot_all_devicecpy.py "$SLURM_JOB_PARTITION" $i
done
