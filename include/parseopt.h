#ifndef PARSEOPT_H
#define PARSEOPT_H

typedef struct parse_option {
	char option;
	int index;
	char *argument;
	struct parse_option *next;
} parse_option;

parse_option *parseopt(int argc, char *argv[], const char *options);

#endif
