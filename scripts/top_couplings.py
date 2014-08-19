#!/usr/bin/env python

import numpy as np
import optparse


def main():
    parser = optparse.OptionParser(usage="%prog [options] coupling_matrix")
    parser.add_option("-s", "--min-separation", type=int, default=7, help="Set the minimum sequence separation of pairs to be outputted [default: %default]")
    parser.add_option("-n", "--num-contacts", type=int, default=30, help="Set the number of pairs to output [default: %default]")

    opt, args = parser.parse_args()

    if len(args) != 1:
        parser.error("Need positional argument!")

    # load coupling matrix
    mat = np.loadtxt(args[0])

    # find top-scoring pairs with sufficient separation
    top = get_top_pairs(mat, opt.num_contacts, opt.min_separation)

    print("#i\tj\tconfidence")
    for i, j, coupling in zip(top[0], top[1], mat[top]):
        print("{0}\t{1}\t{2}".format(i, j, coupling))


def get_top_pairs(mat, num_contacts, min_separation):
    """Get the top-scoring contacts"""

    idx_delta = np.arange(mat.shape[1])[np.newaxis, :] - np.arange(mat.shape[0])[:, np.newaxis]
    mask = idx_delta < min_separation

    mat_masked = np.copy(mat)
    mat_masked[mask] = float("-inf")

    top = mat_masked.argsort(axis=None)[::-1][:(num_contacts)]
    top = (top % mat.shape[0]).astype(np.uint16), np.floor(top / mat.shape[0]).astype(np.uint16)
    return top


if __name__ == '__main__':
    main()
