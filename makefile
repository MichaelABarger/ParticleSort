# Michael Barger (bargerm@cs.pdx.edu)
# makefile for particle sort stuff

CUDACOMP=nvcc
CCOMP=gcc
FLAGS=-arch=sm_11 -g -G

all: particlesortExe filedump
	rm *.o

particlesortExe: particlesortO testharnessO
	$(CUDACOMP) $(FLAGS) particlesortA.o testharness.o -o particlesort

particlesortO:
	$(CUDACOMP) $(FLAGS) -c src/particlesort/particlesortA.cu 

testharnessO:
	$(CUDACOMP) $(FLAGS) -c src/testharness/testharness.cu 

filedump:
	$(CCOMP) src/testutil/filedump.c -o filedump
