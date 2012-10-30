#ifndef TESTHARNESS_H
#define TESTHARNESS_H

#define MAX_BUFFER_SIZE 512

#define EXIT__SUCCESS 0
#define EXIT__ERR -1
#define ERR__NO_INPUT "FATAL ERROR: No input provided!"

extern void TestHarness(void (*)(unsigned short *));
static void FatalError(const char *);
static unsigned int ReadToBuffer(unsigned short *);
static void WriteFromBuffer(const unsigned short *, unsigned int);

#endif
