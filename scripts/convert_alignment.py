#!/usr/bin/env python
"""
Usage:  {prog} infile format outfile
        {prog} infile format
        {prog} format

"""

import Bio.SeqIO


def convert(f_in, format, f_out):
    for record in Bio.SeqIO.parse(f_in, format):
        f_out.write(record.seq.tostring())
        f_out.write("\n")


def main():
    import sys

    fn_in = "-"
    fn_out = "-"
    if len(sys.argv) == 4:
        fn_in, format, fn_out = sys.argv[1:]
    elif len(sys.argv) == 3:
        fn_in, format = sys.argv[1:]
    elif len(sys.argv) == 2:
        format, = sys.argv[1:]
    else:
        sys.stderr.write("Need 1-3 arguments!\n")
        usage()
        sys.exit(1)

    if fn_in == "-":
        f_in = sys.stdin
    else:
        f_in = open(fn_in, "r")

    if fn_out == "-":
        f_out = sys.stdout
    else:
        f_out = open(fn_out, "w")

    convert(f_in, format, f_out)

    if f_in != sys.stdin:
        f_in.close()

    if f_out != sys.stdout:
        f_out.close()


def usage():
    import sys
    print(__doc__.format(prog=sys.argv[0]))

if __name__ == '__main__':
    main()
