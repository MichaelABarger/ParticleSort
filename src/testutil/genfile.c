#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char **argv)
{
	int i;

	if (argc != 2)
		exit(-1);
	for (i = 0; i < atoi(argv[1]); i++) {
		putchar(0);
		putchar(0);
		putchar(0);
		putchar(rand() & 0xff);
	}
}
