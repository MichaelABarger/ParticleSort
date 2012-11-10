#include <stdio.h>
#include "testharness.h"

extern void sort (unsigned int *buffer, unsigned long size)
{
	// nop
}

int main (int argc, char **argv)
{
	unsigned long elapsed = TestHarness(sort);
	fprintf(stderr, "Sort complete; time elapsed: %lu ms\n", elapsed);
	exit(EXIT_SUCCESS);
}
