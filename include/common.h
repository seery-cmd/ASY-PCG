#ifndef COMMON_H

#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <vector>
#include "iostream"
#include "float.h"
using namespace std;
#define VALUE_TYPE_WIDTH 32

#if VALUE_TYPE_WIDTH == 32
    #define MAT_VAL_TYPE float
    #define VALUE_TYPE float
#elif VALUE_TYPE_WIDTH == 64
    #define MAT_VAL_TYPE double
    #define VALUE_TYPE double
#endif

#ifndef MAT_PTR_TYPE

#define MAT_PTR_TYPE int

#endif

#ifndef BENCH_REPEAT

#define BENCH_REPEAT 20

#endif

#ifndef THREAD_NUM

#define THREAD_NUM 4

#endif


// ---------------------------------------
// ---       regular routines      ---
// CSR-SpMV and COO-SpMV, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// CSC-SpMV, CSR-SpTRSV, COO-SpTRSV and CSC-SpTRSV, use 1t, set NTHREADS_MAX = 1
// CSR-SpADD and CSR-SpGEMM, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
//
// ---  inspector-executor routines ---
// CSR-SpMV, COO-SpMV, CSR-SpTRSV and CSC-SpTRSV, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// CSC-SpMV, COO-SpTRSV, use 1t, set NTHREADS_MAX = 1
// CSR-SpADD and CSR-SpGEMM, use 1t 2t 4t 8t 16t 32t 64t, set NTHREADS_MAX = 64
// ---------------------------------------

#ifndef NTHREADS_MAX

#define NTHREADS_MAX 1

#endif

#ifndef NCOL_RHS

#define NCOL_RHS 8

#endif

#ifndef TEST_SPMV

#define TEST_SPMV 0

#endif

#ifndef TEST_SPTRSV

#define TEST_SPTRSV 1

#endif

#ifndef TEST_SPGEMM

#define TEST_SPGEMM 0

#endif

#ifndef STRUCT_NAME_H

#define STRUCT_NAME_H
#define VECTOR_LENGTH 16 / sizeof(VALUE_TYPE)

// CSR 格式存储矩阵
typedef struct sparse_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int* rowptr;
    int* colidx;
    VALUE_TYPE* val;
} sparse_mtx;
typedef struct COO_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int *rowidx;
    int *colptr;
    VALUE_TYPE *val;
}COO_mtx;
typedef struct element_L
{
    int row;
    VALUE_TYPE data;
}element_L;
// ACSR 格式存储矩阵
typedef struct acsr_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int vecnum;
    int* rowptr;
    int* colidx;
    VALUE_TYPE* val;
} acsr_mtx;

typedef struct thread
{
    int coreidx, start, end, tasknum, m, nthreads, threadidx;
    int* rowptr;
    int* colidx;
    VALUE_TYPE* value;
    VALUE_TYPE* Y;
    VALUE_TYPE* X;
} thread;
typedef struct tile_matrix
{
    int *col_index;
    int *row_index;
    VALUE_TYPE *val;
    vector<int> element_array;//偏移量
    int dimension;
    double v;
}tile_matrix;
#endif

#endif