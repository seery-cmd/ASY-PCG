//
// Created by lenovo on 2026/3/2.
//

#ifndef PCG_STRUCT_H
#define PCG_STRUCT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <vector>
#include <algorithm>
#include "iostream"

#define MAT_VAL_TYPE float
#define VALUE_TYPE float
#define MAT_PTR_TYPE int
#define VECTOR_LENGTH 4

typedef struct sparse_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int* rowptr;
    int* colidx;
    VALUE_TYPE* val;
}sparse_mtx;
typedef struct acsr_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int vecnum;
    int* rowptr;
    int* colidx;
    VALUE_TYPE* val;
}acsr_mtx;
typedef struct symmetry_mtx
{
    int rownum;
    int colnum;
    int nnznum;
    int vecnum;//vector数量
    VALUE_TYPE *val_diag;//对角
    int *colIdx_bevel;//斜角联想关联下一个斜角
    int *rowPtr_bevel;
    VALUE_TYPE *val_bevel;
    int *colIdx_horizontal;//水平可以联想到垂直
    int *rowPtr_horizontal;
    VALUE_TYPE *val_horizontal;
    int *colIdx_vertical;//垂直可以联想到水平
    int *rowPtr_vertical;
    VALUE_TYPE *val_vertical;
    //执行顺序,按照先水平-斜角-垂直判断
}symmetry_mtx;
typedef struct sparse_ell
{
    int rowNum;
    int colNum;
    int nnz;
    int tile_num;
    int *colIdx;
    int *tilePtr;
    VALUE_TYPE *val;
}sparse_ell;

#endif //PCG_STRUCT_H
