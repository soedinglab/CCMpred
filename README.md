# CCMpred

[![Travis](https://img.shields.io/travis/soedinglab/CCMpred.svg)](https://travis-ci.org/soedinglab/CCMpred)
[![Codeship](https://img.shields.io/codeship/c2512a40-d488-0132-75f3-623d5159f317.svg)](https://codeship.com/projects/77807)

Protein Residue-Residue **C**ontacts from **C**orrelated **M**utations **pred**icted quickly and accurately.

CCMpred is a C implementation of a Markov Random Field pseudo-likelihood maximization for learning protein residue-residue contacts as made popular by Ekeberg et al. [1] and Balakrishnan and Kamisetty [2]. While predicting contacts with comparable accuracy to the referenced methods, however, CCMpred is written in C / CUDA C, performance-tuned and therefore much faster.

## Requirements

To compile from source, you will need:

  * a recent C compiler (we suggest GCC 4.4 or later)
  * [CMake](http://cmake.org/) 2.8 or later
  * Optional: [NVIDIA CUDA SDK](https://developer.nvidia.com/cuda-downloads) 5.0 or later (if you want to compile for the GPU)

To run CUDA-accelerated computations, you will need an NVIDIA GPU with a Compute Capability of 2.0 or later and the proprietary NVIDIA drivers installed. See the [NVIDIA CUDA GPU Overview](https://developer.nvidia.com/cuda-gpus) for details on your graphics card.

### Memory Requirement on the GPU
When doing computations on the GPU, the available memory limits the size of the model you will be able to compute. We recommend at least 2 GB of GPU RAM so you can calculate contacts for big multiple sequence alignments (e.g for N=5000):

	GPU RAM		Lmax	Lmax(pad)
	1 GB		353	291
	2 GB		512	420
	3 GB		635	519
	5 GB		829	676
	6 GB		911	743
	8 GB		1057	861
	12 GB		1302	1059

You can calculate the memory requirements in bytes for L columns and N rows using the following formula:

	4*(4*(L*L*21*21 + L*20) + 23*N*L + N + L*L) + 2*N*L + 1024

For the padded version:

	4*(4*(L*L*32*21 + L*20) + 23*N*L + N + L*L) + 2*N*L + 1024

## Installation
We recommend compiling CCMpred on the machine that should run the computations so that it can be optimized for the appropriate CPU/GPU architecture.

### Downloading
If you want to compile the most recent version, use the follwing to clone both CCMpred and its submodules:

	git clone --recursive https://github.com/soedinglab/CCMpred.git

### Compilation
With the sourcecode ready, simply run cmake with the default settings and libraries should be auto-detected:

	cmake .
	make

You should find the compiled version of CCMpred at `bin/ccmpred`. To check if the CUDA libraries were detected, you can run `ldd bin/ccmpred` to see if CUDA was linked with the program, or simply run a prediction and check the program's output.

## Useful scripts

The `scripts/` subdirectory contains some python scripts you might find useful - please make sure both NumPy and BioPython are installed to use them!

  * `convert_alignment.py` - Use BioPython's `Bio.SeqIO` to convert a variety of alignment formats (FASTA, etc.) into the CCMpred alignment input format
  * `top_couplings.py` - Extract the top couplings from an output contact maps in a list representation

## License
CCMpred is released under the GNU Affero General Public License v3 or later. See LICENSE for more details.

## References

	[1] Ekeberg, M., Lövkvist, C., Lan, Y., Weigt, M., & Aurell, E. (2013).
	    Improved contact prediction in proteins: Using pseudolikelihoods to infer Potts models.
	    Physical Review E, 87(1), 012707. doi:10.1103/PhysRevE.87.012707

	[2] Balakrishnan, S., Kamisetty, H., Carbonell, J. G., Lee, S.-I., & Langmead, C. J. (2011).
	    Learning generative models for protein fold families.
	    Proteins, 79(4), 1061–78. doi:10.1002/prot.22934
