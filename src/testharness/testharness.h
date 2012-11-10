#ifndef TESTHARNESS_H
#define TESTHARNESS_H

#define MAX_BUFFER_SIZE 512

#define EXIT__SUCCESS 0
#define EXIT__ERR -1
#define ERR__NO_INPUT "FATAL ERROR: No input provided!"

extern unsigned long TestHarness(void (*)(unsigned int *, unsigned long));

#endif
