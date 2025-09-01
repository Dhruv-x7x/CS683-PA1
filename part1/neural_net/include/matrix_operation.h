#ifndef MATRIX_OPERATION_H
#define MATRIX_OPERATION_H

#include "matrix.h"
#include "defines.h"

extern int counter;

class MatrixOperation {
	public:
		//const constexpr Tile_Size = TILE_SIZE;
	
	// Naive Matrix Multiplication
	static Matrix NaiveMatMul(const Matrix &A, const Matrix &B);
	// Reordered Matrix Multiplication
	static Matrix ReorderedMatMul(const Matrix &A, const Matrix &B);
	// Unrolled Matrix Multiplication
	static Matrix UnrolledMatMul(const Matrix &A, const Matrix &B);
	// Tiled Matrix Multiplication
	static Matrix TiledMatMul(const Matrix &A, const Matrix &B);
	// Vectorized Matrix Multiplication
	static Matrix VectorizedMatMul(const Matrix &A, const Matrix &B);

	// Matrix transpose
	static Matrix Transpose(const Matrix &A);

	// Matrix multiplication dispatcher
	static Matrix MatMul(const Matrix &A, const Matrix &B, MatrixOptimization opt ) {
		counter++;
		switch (opt) {
			case NAIVE:
				return NaiveMatMul(A, B);
			case REORDERED:
				return ReorderedMatMul(A, B);
			case UNROLLED:
				return UnrolledMatMul(A, B);
			case TILED:
				return TiledMatMul(A, B);
			case VECTORIZED:
				return VectorizedMatMul(A, B);
			default:
				throw std::invalid_argument("Invalid matrix multiplication optimization option.");
		}
	}

};


#endif // MATRIX_OPERATION_H
