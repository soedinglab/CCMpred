#include <stdlib.h>
#include <stdio.h>
#include "parseopt.h"


/**
 * Constructor for parse_option objects
 */
parse_option *new_parse_option(char option, char* argument, int index) {
	parse_option *opt = (parse_option *)malloc(sizeof(parse_option));
	if(opt == NULL) {
		perror("Cannot malloc option!");
	}

	opt->option = option;
	opt->argument = argument;
	opt->index = index;
	opt->next = NULL;

	return opt;
}

/**
 * Find the index of an option character opt in the options definition options
 */
int findopt(char opt, const char* options) {
	int opti;
	opti = 0;

	while((options[opti] != 0) && (options[opti] != opt)) { opti++; }
	return opti;
}

/**
 * Parse an array of command line arguments
 */
parse_option *parseopt(int argc, char *argv[], const char *options) {
	int argi, argii, opti;
	char *arg;

	parse_option *head, *tail, *newopt;

	argi = 1;

	head = NULL;
	tail = NULL;

	
	while(argi < argc) {
		arg = argv[argi];

		if(arg[0] == '-') {
			/* Handle option */

			argii = 1;

			while(arg[argii] != 0) {

				opti = findopt(arg[argii], options);
				if(options[opti] == 0) {
					printf("Unknown option %c at argument index %d\n", arg[argii], opti);
					return NULL;
				}

				if(options[opti+1] == ':') {
					/* The current option expects an argument */

					if(arg[argii+1] == 0) {
						/* Argument is separate token */

						if(argi + 1 >= argc) {
							printf("Option %c expected an argument but encountered end of command line!\n", arg[argii]);
							return NULL;
						}

						newopt = new_parse_option(options[opti], argv[argi+1], argi);
						argi++;
					} else {
						/* Argument is fused with option */
						newopt = new_parse_option(options[opti], &(arg[argii+1]), argi);
						while(arg[argii+1] != 0) { argii++; }
					}

				} else {
					/* The current option expects no argument */
					newopt = new_parse_option(options[opti], NULL, argi);
				}

				if(head == NULL) {
					head = tail = newopt;
				} else {
					tail->next = newopt;
					tail = newopt;
				}

				argii++;
			}

		} else {
			/* Handle positional argument */

			newopt = new_parse_option(0, argv[argi], argi);

			if(head == NULL) {
				head = tail = newopt;
			} else {
				tail->next = newopt;
				tail = newopt;
			}

		}

		argi++;
	}


	return head;
}

