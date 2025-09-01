/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes

#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX
#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	16		// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            double a_ik = A[i * size + k];
            int j = 0;

            // Loop unrolled by 8
            for (; j <= size - 8; j += 8) {
                C[i * size + j]     += a_ik * B[k * size + j];
                C[i * size + j + 1] += a_ik * B[k * size + j + 1];
                C[i * size + j + 2] += a_ik * B[k * size + j + 2];
                C[i * size + j + 3] += a_ik * B[k * size + j + 3];
                C[i * size + j + 4] += a_ik * B[k * size + j + 4];
                C[i * size + j + 5] += a_ik * B[k * size + j + 5];
                C[i * size + j + 6] += a_ik * B[k * size + j + 6];
                C[i * size + j + 7] += a_ik * B[k * size + j + 7];
            }

            // Leftover elements
            for (; j < size; j++) {
                C[i * size + j] += a_ik * B[k * size + j];
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int ii = 0; ii < size; ii += tile_size) {
        for (int jj = 0; jj < size; jj += tile_size) {
            for (int kk = 0; kk < size; kk += tile_size) {

                int i_max = (ii + tile_size > size) ? size : ii + tile_size;
                int j_max = (jj + tile_size > size) ? size : jj + tile_size;
                int k_max = (kk + tile_size > size) ? size : kk + tile_size;

                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = 0.0;
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i * size + k] * B[k * size + j];
                        }
                        C[i * size + j] += sum;
                    }
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

#pragma GCC push_options
#pragma GCC target("avx,fma,sse2")
/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 4) {
            __m256d c_vec = _mm256_loadu_pd(&C[i * size + j]);
            for (int k = 0; k < size; k++) {
                __m256d a_ik = _mm256_set1_pd(A[i * size + k]);
                __m256d b_vec = _mm256_loadu_pd(&B[k * size + j]);
                c_vec = _mm256_fmadd_pd(a_ik, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C[i * size + j], c_vec);
        }
        for (int j = size - (size % 4); j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] += sum;
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
}
#pragma GCC pop_options
/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/



#pragma GCC push_options
#pragma GCC target("avx,fma,sse2")
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
    for (int ii = 0; ii < size; ii += tile_size) {
        for (int kk = 0; kk < size; kk += tile_size) {
            for (int jj = 0; jj < size; jj += tile_size) {
                for (int i = ii; i < ii + tile_size && i < size; i++) {
                    for (int k = kk; k < kk + tile_size && k < size; k++) {
                        __m256d a_vec = _mm256_set1_pd(A[i * size + k]);
                        int j = jj;
                        for (; j + 7 < jj + tile_size && j + 7 < size; j += 8) {
                            __m256d c0 = _mm256_loadu_pd(&C[i * size + j]);
                            __m256d c1 = _mm256_loadu_pd(&C[i * size + j + 4]);
                            __m256d b0 = _mm256_loadu_pd(&B[k * size + j]);
                            __m256d b1 = _mm256_loadu_pd(&B[k * size + j + 4]);
                            c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                            c1 = _mm256_fmadd_pd(a_vec, b1, c1);
                            _mm256_storeu_pd(&C[i * size + j], c0);
                            _mm256_storeu_pd(&C[i * size + j + 4], c1);
                        }
                        for (; j < jj + tile_size && j < size; j++) {
                            C[i * size + j] += A[i * size + k] * B[k * size + j];
                        }
                    }
                }
            }
        }
    }
}

#pragma GCC pop_options

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
