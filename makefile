# Michael Barger (bargerm@cs.pdx.edu)
# makefile for particle sort stuff

CUDACOMP=nvcc
CCOMP=gcc
FLAGS=-arch=sm_11 -g -G

all: particlesortExe filedump genfile
	rm *.o

particlesortExe: particlesortAO particlesortBO particlesortCO testharnessO
	$(CUDACOMP) $(FLAGS) particlesortA.o testharness.o -o particlesortA
	$(CUDACOMP) $(FLAGS) particlesortB.o testharness.o -o particlesortB
	$(CUDACOMP) $(FLAGS) particlesortC.o testharness.o -o particlesortC

particlesortAO:
	$(CUDACOMP) $(FLAGS) -c src/particlesort/particlesortA.cu 
	
particlesortBO:
	$(CUDACOMP) $(FLAGS) -c src/particlesort/particlesortB.cu 

particlesortCO:
	$(CUDACOMP) $(FLAGS) -c src/particlesort/particlesortC.cu 

testharnessO:
	$(CUDACOMP) $(FLAGS) -c src/testharness/testharness.cu 

filedump:
	$(CCOMP) src/testutil/filedump.c -o filedump

genfile:
	$(CCOMP) src/testutil/genfile.c -o genfile
