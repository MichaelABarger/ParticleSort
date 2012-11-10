#ifndef TEST_HARNESS_C
#define TEST_HARNESS_C

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include "testharness.h"

static void FatalError(const char *);
static unsigned long ReadToBuffer(unsigned int *);
static void WriteFromBuffer(const unsigned int *, unsigned long);

extern unsigned long TestHarness(void (*sort)(unsigned int *, unsigned long))
{
	struct timeval tv;
	unsigned long starttime, endtime, elapsed;
	unsigned int *buffer = (unsigned int *)malloc(MAX_BUFFER_SIZE * sizeof(int));

	/* start timer */
	gettimeofday(&tv, NULL);
	starttime = tv.tv_sec * 1000 + tv.tv_usec / 1000;

	fprintf(stderr, "Reading from standard input\n");
	unsigned int buffer_size = ReadToBuffer(buffer);
	if (buffer_size < 1)
		FatalError(ERR__NO_INPUT);

	fprintf(stderr, "** SORTING **\n");
	sort(buffer, buffer_size);

	fprintf(stderr, "Writing %u charaters to output\n", buffer_size);
	WriteFromBuffer(buffer, buffer_size);
	
	/* end timer */
	gettimeofday(&tv, NULL);
	endtime = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	elapsed = endtime - starttime;
	
	free(buffer);
	return elapsed;
}

/* Note that although the two functions below write to / read from unsigned int arrays,
   the values they are reading and writing are at most two bytes long (unsigned short);
   this is on purpose. */
static unsigned long ReadToBuffer(unsigned int *buffer)
{
	int i;
	unsigned int c1, c2, c3, c4;

	for (i = 0; i < MAX_BUFFER_SIZE; i++) {
		c1 = getchar();
		if (feof(stdin))
			break;
		c2 = getchar();
		if (feof(stdin))
			break;
		c3 = getchar();
		if (feof(stdin))
			break;
		c4 = getchar();
		if (feof(stdin))
			break;
		*(buffer++) = (c1 << 24) | (c2 << 16) | (c3 << 8) | c4;
	}
	return i;
}

static void WriteFromBuffer(const unsigned int *buffer, unsigned long buffer_size)
{
	int i;

	for (i = 0; i < buffer_size; i++) {
		unsigned int val = *(buffer++);
		putchar((val >> 24) & 0xff);
		putchar((val >> 16) & 0xff);
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
