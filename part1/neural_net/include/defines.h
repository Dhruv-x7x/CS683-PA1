#ifndef DEFINES_H
#define DEFINES_H

#define INITIAL_VALUE 0.0
#define TILE_SIZE 16 // Size of the tile for tiled matrix multiplication


//Data type of each element in the matrix
typedef double element_t;

enum MatrixOptimization {
	NAIVE, // Basic triple loop implementation
	REORDERED, 
	UNROLLED, 
	TILED,
	VECTORIZED
};


#endif // DEFINES_H

