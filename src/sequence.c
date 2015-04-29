
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>


#define SEQ_BUFFER 8192

const unsigned char AMINO_INDICES[26] = {
	//  A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z
	    0, 20,  4,  3,  6, 13,  7,  8,  9, 20, 11, 10, 12,  2, 20, 14,  5,  1, 15, 16, 20, 19, 17, 20, 18, 20 };

const unsigned char CHAR_INDICES[21] = {
	//  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
	//  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   -
	   65, 82, 78, 68, 67, 81, 69, 71, 72, 73, 76, 75, 77, 70, 80, 83, 84, 87, 89, 86, 45 };

/** Convert amino acid ASCII code to an index, with 0 = gap and 1 = A, 2 = C, etc.
 */
unsigned char aatoi(unsigned char aa) {
	if(!isalpha(aa)) { return 20; }

	aa = toupper(aa);
	if(aa < 65 || aa > 90) { return 20; }
	return AMINO_INDICES[aa - 65];
}

unsigned char itoaa(unsigned char i) {
	return CHAR_INDICES[i];
}

void trim_right(char *str) {

	// find first non-whitespace character from right
	char *end = str + strlen(str) - 1;
	while(end > str && isspace(*end)) end--;

	// add new null terminator
	*(end+1) = 0;
}

/** Read an MSA file into an index matrix.
 *
 * Matrix will be returned as a pointer to a one-dimensional int array
 * where the cell at row i and column j will have the index i + nrow*j
 */
unsigned char* read_msa(FILE *f, int *ncol, int *nrow) {
	char buf[SEQ_BUFFER];
	int nc;

	*nrow = 0;
	*ncol = 0;
	while( fgets(buf, SEQ_BUFFER, f) ) {
		(*nrow)++;

		trim_right(buf);

		if(strstr(buf, ">")) {
			printf("ERROR: The alignment file seems to be in a3m or fasta format!\nPlease reformat the alignment to PSICOV format!\n\n");
			exit(100);
		}

		nc = strlen(buf);
		*ncol = nc > *ncol ? nc : *ncol;
	}

	unsigned char* out = (unsigned char*)malloc(sizeof(unsigned char) * ( *ncol * *nrow));

	rewind(f);

	for(int i=0; i < *nrow; i++) {
		fgets(buf, SEQ_BUFFER, f);
		for(int j = 0; j < *ncol; j++) {
			out[i * *ncol + j] = aatoi( buf[j] );
		}
	}

	return out;
}
