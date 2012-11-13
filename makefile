# Michael Barger (bargerm@cs.pdx.edu)
# makefile for particle sort stuff

CUDACOMP=nvcc
CCOMP=gcc
FLAGS=-arch=sm_11 -O1 -Xopencc -O1 -Xptxas -O1

all: particlesortExe testharnesstest filedump genfile emptykernel particlesort1block
	rm *.o

particlesortExe: particlesortO testharnessO
	$(CUDACOMP) $(FLAGS) particlesort.o testharness.o -o particlesort

particlesort1block: testharnessO
	$(CUDACOMP) $(FLAGS) src/particlesort/particlesort-1block.cu testharness.o -o particlesort-1block

emptykernel: testharnessO
	$(CUDACOMP) $(FLAGS) src/particlesort/emptykernel.cu testharness.o -o emptykernel

particlesortO:
	$(CUDACOMP) $(FLAGS) -c src/particlesort/particlesort.cu 

testharnesstest: testharnessO
	$(CUDACOMP) $(FLAGS) testharness.o src/testharness/testharnesstest.cu -o testharnesstest
	
testharnessO:
	$(CUDACOMP) $(FLAGS) -c src/testharness/testharness.cu 

filedump:
	$(CCOMP) src/testutil/filedump.c -o filedump

genfile:
	$(CCOMP) src/testutil/genfile.c -o genfile
