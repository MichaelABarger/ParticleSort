#ifndef TEST_HARNESS_C
#define TEST_HARNESS_C

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include "testharness.h"

extern void TestHarness(void (*sort)(unsigned short *))
{
	struct timeval tv;
	unsigned long starttime, endtime, elapsed;
	unsigned short *buffer = (unsigned short *)malloc(MAX_BUFFER_SIZE * 2);

	gettimeofday(&tv, NULL);
	starttime = tv.tv_sec * 1000 + tv.tv_usec / 1000;

	fprintf(stderr, "Reading from standard input:\n");
	unsigned int buffer_size = ReadToBuffer(buffer);
	if (buffer_size < 1)
		FatalError(ERR__NO_INPUT);

	fprintf(stderr, "** SORTING **\n");
	sort(buffer);

	fprintf(stderr, "Writing %u charaters to output:\n", buffer_size);
	WriteFromBuffer(buffer, buffer_size);
	
	gettimeofday(&tv, NULL);
	endtime = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	elapsed = endtime - starttime;
	fprintf(stderr, "%lu milliseconds elapsed.\n", elapsed);
	
	free(buffer);
	exit(EXIT__SUCCESS);
}

static unsigned int ReadToBuffer(unsigned short *buffer)
{
	int i;
	char c1, c2;

	for (i = 0; i < MAX_BUFFER_SIZE; i++) {
		c1 = getchar();
		if (feof(stdin))
			break;
		c2 = getchar();
		if (feof(stdin))
			break;
		*(buffer++) = (unsigned short)((c2 << 8) | c1);
	}
	return (i > 0) ? i : 0;
}

static void WriteFromBuffer(const unsigned short *buffer, unsigned int buffer_size)
{
	int i;

	for (i = 0; i < buffer_size; i++) {
		unsigned short val = *(buffer++);
		putchar((val >> 8) & 0xff);
		putchar(val & 0xff);
	}
}

static void FatalError(const char *msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(EXIT__ERR);
}

#endif
