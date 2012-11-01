#include <stdio.h>

int main(int argc, char **argv)
{
	int c1, c2;
	int i = 0;

	while (1) {
		c1 = getchar();
		if (feof(stdin))
			break;
		c2 = getchar();
		if (feof(stdin))
			break;
		unsigned short word = (c1 << 8) | c2;
		printf("    %6u, ", word);
		
		if ((++i & 7) == 0)
			printf("\n");
	}
	printf("\n");
	return 0;
}
