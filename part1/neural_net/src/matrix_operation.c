#include "matrix_operation.h"
#include <immintrin.h>

#pragma GCC push_options
#pragma GCC target("avx,fma")
Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(size_t i = 0; i < n ; i++) {
		for (size_t j = 0 ; j< m ; j++) {
			for(size_t l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
		for (size_t k2 = 0; k2 < k; ++k2) {
			element_t r = A(i, k2);
			for (size_t j = 0; j < m; ++j) {
				C(i, j) += r * B(k2, j);
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 8;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
	for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            element_t sum = 0;
            size_t kk = 0;
            for (; kk + UNROLL <= k; kk += UNROLL) {
                for (int u = 0; u < UNROLL; ++u) {
                    sum += A(i, kk + u) * B(kk + u, j);
                }
            }
            C(i, j) = sum;
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 64;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t ii = 0; ii < n; ii += T) {
        for (size_t kk = 0; kk < k; kk += T) {
            for (size_t jj = 0; jj < m; jj += T) {
                for (size_t i = ii; i < ii + T; ++i) {
                    for (size_t kk2 = kk; kk2 < kk + T; ++kk2) {
                        auto a = A(i, kk2);
                        for (size_t j = jj; j < jj + T; ++j) {
                            C(i, j) += a * B(kk2, j);
                        }
                    }
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
	for (size_t i = 0; i < n; ++i) {
        size_t j = 0;
        for (; j + 3 < m; j += 4) {
            __m256d c_vec = _mm256_loadu_pd(&C(i, j));
            for (size_t kk = 0; kk < k; ++kk) {
                __m256d a_vec = _mm256_set1_pd(A(i, kk));
                __m256d b_vec = _mm256_loadu_pd(&B(kk, j));
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C(i, j), c_vec);
        }
        for (; j < m; ++j) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                sum += A(i, kk) * B(kk, j);
            }
            C(i, j) += sum;
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	/*for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j, i) = A(i, j);
		}
	}*/

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    size_t blockSize = 16;
    for (size_t ii = 0; ii < rows; ii += blockSize) {
        for (size_t jj = 0; jj < cols; jj += blockSize) {
            for (size_t i = ii; i < ii + blockSize; ++i) {
                for (size_t j = jj; j < jj + blockSize; ++j) {
                    result(j, i) = A(i, j);
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}
