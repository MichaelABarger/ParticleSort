#ifndef TEST_HARNESS_C
#define TEST_HARNESS_C

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include "testharness.h"

extern unsigned long TestHarness(void (*sort)(unsigned short *, unsigned long))
{
	struct timeval tv;
	unsigned long starttime, endtime, elapsed;
	unsigned short *buffer = (unsigned short *)malloc(MAX_BUFFER_SIZE * 2);

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
static unsigned long ReadToBuffer(unsigned short *buffer)
{
	int i;
	int c1, c2;

	for (i = 0; i < MAX_BUFFER_SIZE; i++) {
		c1 = getchar();
		if (feof(stdin))
			break;
		c2 = getchar();
		if (feof(stdin))
			break;
		*(buffer++) = (unsigned int)((c1 << 8) | c2) & 0xffff;
	}
	return i;
}

static void WriteFromBuffer(const unsigned short *buffer, unsigned long buffer_size)
{
	int i;

	for (i = 0; i < buffer_size; i++) {
		unsigned int val = *(buffer++);
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
