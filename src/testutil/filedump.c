#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv)
{
	unsigned int c1, c2, c3, c4;
	int i = 0;

	while (1) {
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
		unsigned int word = (c1 << 24) | (c2 << 16) | (c3 << 8) | c4;
		printf(" %8u, ", word);
		
		if ((++i & 7) == 0)
			printf("\n");
	}
	printf("\n");
	return 0;
}
