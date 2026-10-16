#ifndef _UTILS_
#define _UTILS_

#include "../struct/struct.h"
#include "arm_neon.h"

void Print_csr(sparse_mtx *mtx)
{
    printf("-----------------------row Ptr--------------------------\n");
    for(int i=0;i<mtx->rownum;i++)
    {
        printf("row%d==%d,%d\n",i,mtx->rowptr[i],mtx->rowptr[i+1]);
    }
    printf("------------------------col-Idx--------------------------\n");
    for(int i=0;i<mtx->rownum;i++)
    {
        for(int j=mtx->rowptr[i];j<mtx->rowptr[i+1];j++)
        {
            printf("%d ",mtx->colidx[j]);
        }
        printf("\n");
    }
    printf("------------------------val--------------------------\n");
    for(int i=0;i<mtx->rownum;i++)
    {
        for(int j=mtx->rowptr[i];j<mtx->rowptr[i+1];j++)
        {
            printf("%lf ",mtx->val[j]);
        }
        printf("\n");
    }
}
// in-place exclusive scan
void exclusive_scan(MAT_PTR_TYPE* input, int length)
{
    if(length == 0 || length == 1) return;

    MAT_PTR_TYPE old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}
acsr_mtx* CSR2ACSR(sparse_mtx* mtx)
{
    acsr_mtx* res = (acsr_mtx*)malloc(sizeof(acsr_mtx));
    res->rownum = mtx->rownum;
    res->colnum = mtx->colnum;
    res->nnznum = mtx->nnznum;
    res->rowptr = (int*)malloc((mtx->rownum+1) * sizeof(int));

    res->rowptr[mtx->rownum] = 0;
    for (int row = 0; row < mtx->rownum; ++row)
    {
        int start_idx = mtx->rowptr[row];
        int end_idx = mtx->rowptr[row + 1];
        int vector_start = start_idx;
        int vector_num = 0,flag = 0;
        for(int index = start_idx; index < end_idx; ++index)
        {
            flag = 1;
            if(mtx->colidx[index] - mtx->colidx[vector_start] >= VECTOR_LENGTH)
            {
                vector_num++;
                vector_start = index;
            }
        }
        if(flag)    vector_num++;
        flag = 0;
        res->rowptr[row] = vector_num;
        res->rowptr[mtx->rownum] += vector_num; // 这里 rowptr 并没有保存行偏移，而是保存的每行的向量个数
    }
    res->vecnum = res->rowptr[mtx->rownum];
    res->colidx = (int*)malloc(res->vecnum * sizeof(int));  // 有多少个向量就申请多少空间
    res->val = (VALUE_TYPE*)malloc((res->vecnum * VECTOR_LENGTH) * sizeof(VALUE_TYPE));
    memset(res->colidx, 0, (res->vecnum) * sizeof(int));
    memset(res->val , 0, (res->vecnum * VECTOR_LENGTH) * sizeof(VALUE_TYPE));
    int vector_num = 0, flag;
    for (int row = 0; row < mtx->rownum; ++row)
    {
        flag = 0;
        int start_idx = mtx->rowptr[row];
        int end_idx = mtx->rowptr[row + 1];
        int vector_start = start_idx;
        for (int index = start_idx; index < end_idx; ++index)
        {
            flag = 1;
            if (mtx->colidx[index] - mtx->colidx[vector_start] >= VECTOR_LENGTH)
            {
                vector_start = index;
                ++vector_num;
            }
            int vector_validx = mtx->colidx[index] - mtx->colidx[vector_start];
            res->colidx[vector_num] = mtx->colidx[vector_start];
            res->val[vector_num * VECTOR_LENGTH + vector_validx] = mtx->val[index];
        }
        if (flag)
        {
            vector_num++;
        }

    }
    exclusive_scan(res->rowptr, mtx->rownum+1);
    return res;
}
void printSparseMatrix(sparse_mtx* mtx)
{
    int rowIndex = 0;
    int nzIndex = 0;

    for (rowIndex = 0; rowIndex < mtx->rownum; rowIndex++)
    {
        for (int colIndex = 0; colIndex < mtx->colnum; colIndex++)
        {
            if (colIndex == mtx->colidx[nzIndex] && nzIndex < mtx->rowptr[rowIndex + 1])
            {
                printf("%lg ", mtx->val[nzIndex]);
                nzIndex++;
            }
            else
            {
                printf("* ");
            }
        }
        printf("\n");
    }
}
void trans_symmetry_mtx(sparse_mtx *mtx,symmetry_mtx *sy_mtx) {
    sy_mtx->rownum = mtx->rownum;
    sy_mtx->colnum = mtx->colnum;
    sy_mtx->nnznum = mtx->nnznum;
    sy_mtx->val_diag = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->colnum + 1) / 2 * 2);//需要补齐最后一个元素
    //斜角
    sy_mtx->colIdx_bevel = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_bevel = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_bevel = (int *) malloc(sizeof(int) * (sy_mtx->rownum / 2 + 1));
    sy_mtx->rowPtr_bevel[0] = 0;
    //垂直
    sy_mtx->colIdx_vertical = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_vertical = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_vertical = (int *) malloc(sizeof(int) * (sy_mtx->rownum / 2 + 1));
    sy_mtx->rowPtr_vertical[0] = 0;
    //水平
    sy_mtx->colIdx_horizontal = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_horizontal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_horizontal = (int *) malloc(sizeof(int) * (sy_mtx->rownum + 1));
    sy_mtx->rowPtr_horizontal[0] = 0;

    int *switch_A = (int *) malloc(sizeof(int) * sy_mtx->colnum);

    int b = sy_mtx->rownum / 2;
    int i, j, z;
    int beId, verId, horId, diagId;
    beId = verId = horId = diagId = 0;
    int vecNum = 0;
    int abs;
    //tow row in the loop to collect vectors
    for (i = 0; i < b; i++) {
        //init the next row colIdx
        for (j = 0; j < sy_mtx->colnum; j++) {
            switch_A[j] = 0;
        }
        //first row
        for (j = mtx->rowptr[i * 2]; j < mtx->rowptr[i * 2 + 1]; j++) {
            abs = 0;
            //diag以上
            if (mtx->colidx[j] < i * 2) {
                continue;
            } else if (mtx->colidx[j] == i * 2) {
                sy_mtx->val_diag[diagId] = mtx->val[j];
                diagId++;
                continue;
            }
            //水平>斜角>垂直判断
            else if (j + 1 < mtx->rowptr[i * 2 + 1] && (mtx->colidx[j + 1] - mtx->colidx[j]) < 2) {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = mtx->val[j + 1];
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                j++;
                vecNum++;
                continue;
            }
            //斜角
            for (z = mtx->rowptr[i * 2 + 1]; z < mtx->rowptr[(i + 1) * 2]; z++) {
                if (mtx->colidx[z] == mtx->colidx[j] + 1 && switch_A[mtx->colidx[z]] != 1) {
                    sy_mtx->val_bevel[beId * 2] = mtx->val[j];
                    sy_mtx->val_bevel[beId * 2 + 1] = mtx->val[z];
                    sy_mtx->colIdx_bevel[beId] = mtx->colidx[j];
                    beId++;
                    vecNum++;
                    abs = 1;
                    switch_A[mtx->colidx[z]] = 1;
                    break;
                }
            }
            if (abs) {
                continue;
            }
            //垂直
            for (z = mtx->rowptr[i * 2 + 1]; z < mtx->rowptr[(i + 1) * 2]; z++) {
                if (mtx->colidx[z] == mtx->colidx[j] && switch_A[mtx->colidx[z]] != 1 && mtx->colidx[z] != i * 2 + 1) {
                    sy_mtx->val_vertical[verId * 2] = mtx->val[j];
                    sy_mtx->val_vertical[verId * 2 + 1] = mtx->val[z];
                    sy_mtx->colIdx_vertical[verId] = mtx->colidx[j];
                    verId++;
                    vecNum++;
                    abs = 1;
                    switch_A[mtx->colidx[z]] = 1;
                    break;
                }
            }
            if (abs) {
                continue;
            }
            sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
            sy_mtx->val_horizontal[horId * 2 + 1] = 0.0f;
            sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
            horId++;
            vecNum++;
        }
        sy_mtx->rowPtr_bevel[i + 1] = beId;
        sy_mtx->rowPtr_vertical[i + 1] = verId;
        sy_mtx->rowPtr_horizontal[i * 2 + 1] = horId;
        //next row forever exist
        for (j = mtx->rowptr[i * 2 + 1]; j < mtx->rowptr[(i + 1) * 2]; j++) {
            if (mtx->colidx[j] < i * 2 + 1 || switch_A[mtx->colidx[j]] == 1) {
                continue;
            } else if (mtx->colidx[j] == i * 2 + 1) {
                sy_mtx->val_diag[diagId] = mtx->val[j];
                diagId++;
                continue;
            }else if (j + 1 < mtx->rowptr[(i + 1) * 2] && (mtx->colidx[j + 1] - mtx->colidx[j]) < 2 &&
                       switch_A[mtx->colidx[j + 1]] == 0) {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = mtx->val[j + 1];
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                j++;
                vecNum++;
                continue;
            } else {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = 0.0f;
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                vecNum++;
                continue;
            }
        }
        sy_mtx->rowPtr_horizontal[(i + 1) * 2] = horId;
    }
    sy_mtx->vecnum = vecNum + sy_mtx->rownum/2;
    //diag add
    if (sy_mtx->rownum & 1) {
        sy_mtx->val_diag[mtx->rownum - 1] = mtx->val[mtx->nnznum - 1];
        sy_mtx->val_diag[mtx->rownum] = 0.0f;
    }
    free(switch_A);
}
void trans_asymmetry_mtx(sparse_mtx *mtx,symmetry_mtx *sy_mtx) {
    sy_mtx->rownum = mtx->rownum;
    sy_mtx->colnum = mtx->colnum;
    sy_mtx->nnznum = mtx->nnznum;
    //斜角
    sy_mtx->colIdx_bevel = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_bevel = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_bevel = (int *) malloc(sizeof(int) * (sy_mtx->rownum / 2 + 1));
    sy_mtx->rowPtr_bevel[0] = 0;
    //垂直
    sy_mtx->colIdx_vertical = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_vertical = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_vertical = (int *) malloc(sizeof(int) * (sy_mtx->rownum / 2 + 1));
    sy_mtx->rowPtr_vertical[0] = 0;
    //水平
    sy_mtx->colIdx_horizontal = (int *) malloc(sizeof(int) * (sy_mtx->nnznum) * 2);
    sy_mtx->val_horizontal = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * (sy_mtx->nnznum) * 2);
    sy_mtx->rowPtr_horizontal = (int *) malloc(sizeof(int) * (sy_mtx->rownum + 1));
    sy_mtx->rowPtr_horizontal[0] = 0;

    int *switch_A = (int *) malloc(sizeof(int) * sy_mtx->colnum);

    int b = sy_mtx->rownum / 2;
    int i, j, z;
    int beId, verId, horId, diagId;
    beId = verId = horId = diagId = 0;
    int vecNum = 0;
    int abs;
    //tow row in the loop to collect vectors
    for (i = 0; i < b; i++) {
        //init the next row colIdx
        for (j = 0; j < sy_mtx->colnum; j++) {
            switch_A[j] = 0;
        }
        //first row
        for (j = mtx->rowptr[i * 2]; j < mtx->rowptr[i * 2 + 1]; j++) {
            abs = 0;
            if (j + 1 < mtx->rowptr[i * 2 + 1] && (mtx->colidx[j + 1] - mtx->colidx[j]) < 2) {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = mtx->val[j + 1];
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                j++;
                vecNum++;
                continue;
            }
            //斜角
            for (z = mtx->rowptr[i * 2 + 1]; z < mtx->rowptr[(i + 1) * 2]; z++) {
                if (mtx->colidx[z] == mtx->colidx[j] + 1 && switch_A[mtx->colidx[z]] != 1) {
                    sy_mtx->val_bevel[beId * 2] = mtx->val[j];
                    sy_mtx->val_bevel[beId * 2 + 1] = mtx->val[z];
                    sy_mtx->colIdx_bevel[beId] = mtx->colidx[j];
                    beId++;
                    vecNum++;
                    abs = 1;
                    switch_A[mtx->colidx[z]] = 1;
                    break;
                }
            }
            if (abs) {
                continue;
            }
            //垂直
            for (z = mtx->rowptr[i * 2 + 1]; z < mtx->rowptr[(i + 1) * 2]; z++) {
                if (mtx->colidx[z] == mtx->colidx[j] && switch_A[mtx->colidx[z]] != 1) {
                    sy_mtx->val_vertical[verId * 2] = mtx->val[j];
                    sy_mtx->val_vertical[verId * 2 + 1] = mtx->val[z];
                    sy_mtx->colIdx_vertical[verId] = mtx->colidx[j];
                    verId++;
                    vecNum++;
                    abs = 1;
                    switch_A[mtx->colidx[z]] = 1;
                    break;
                }
            }
            if (abs) {
                continue;
            }
            sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
            sy_mtx->val_horizontal[horId * 2 + 1] = 0.0f;
            sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
            horId++;
            vecNum++;
        }
        sy_mtx->rowPtr_bevel[i + 1] = beId;
        sy_mtx->rowPtr_vertical[i + 1] = verId;
        sy_mtx->rowPtr_horizontal[i * 2 + 1] = horId;
        //next row forever exist
        for (j = mtx->rowptr[i * 2 + 1]; j < mtx->rowptr[(i + 1) * 2]; j++) {
            if (switch_A[mtx->colidx[j]] == 1) {
                continue;
            }
            else if (j + 1 < mtx->rowptr[(i + 1) * 2] && (mtx->colidx[j + 1] - mtx->colidx[j]) < 2 &&
                      switch_A[mtx->colidx[j + 1]] == 0) {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = mtx->val[j + 1];
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                j++;
                vecNum++;
                continue;
            } else {
                sy_mtx->val_horizontal[horId * 2] = mtx->val[j];
                sy_mtx->val_horizontal[horId * 2 + 1] = 0.0f;
                sy_mtx->colIdx_horizontal[horId] = mtx->colidx[j];
                horId++;
                vecNum++;
                continue;
            }
        }
        sy_mtx->rowPtr_horizontal[(i + 1) * 2] = horId;
    }
    sy_mtx->vecnum = vecNum;
    free(switch_A);
}
void Print_syMtx(symmetry_mtx *mtx)
{
    printf("------------------------diag--------------------------\n");
    for(int i=0;i<(mtx->rownum+1)/2*2;i++)
    {
        printf("%lf\n",mtx->val_diag[i]);
    }
    printf("------------------------水平--------------------------\n");
    for(int i=0;i<mtx->rownum;i++)
    {
        for(int j=mtx->rowPtr_horizontal[i];j<mtx->rowPtr_horizontal[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_horizontal[j],mtx->val_horizontal[j*2],mtx->val_horizontal[j*2+1]);
        }
        printf("\n");
    }
    printf("------------------------斜角--------------------------\n");
    for(int i=0;i<mtx->rownum / 2;i++)
    {
        for(int j=mtx->rowPtr_bevel[i];j<mtx->rowPtr_bevel[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_bevel[j],mtx->val_bevel[j*2],mtx->val_bevel[j*2+1]);
        }
        printf("\n");
    }
    printf("------------------------垂直--------------------------\n");
    for(int i=0;i<mtx->rownum / 2;i++)
    {
        for(int j=mtx->rowPtr_vertical[i];j<mtx->rowPtr_vertical[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_vertical[j],mtx->val_vertical[j*2],mtx->val_vertical[j*2+1]);
        }
        printf("\n");
    }
}
void Print_asyMtx(symmetry_mtx *mtx)
{
    printf("------------------------水平--------------------------\n");
    for(int i=0;i<mtx->rownum;i++)
    {
        for(int j=mtx->rowPtr_horizontal[i];j<mtx->rowPtr_horizontal[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_horizontal[j],mtx->val_horizontal[j*2],mtx->val_horizontal[j*2+1]);
        }
        printf("\n");
    }
    printf("------------------------斜角--------------------------\n");
    for(int i=0;i<mtx->rownum / 2;i++)
    {
        for(int j=mtx->rowPtr_bevel[i];j<mtx->rowPtr_bevel[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_bevel[j],mtx->val_bevel[j*2],mtx->val_bevel[j*2+1]);
        }
        printf("\n");
    }
    printf("------------------------垂直--------------------------\n");
    for(int i=0;i<mtx->rownum / 2;i++)
    {
        for(int j=mtx->rowPtr_vertical[i];j<mtx->rowPtr_vertical[i+1];j++)
        {
            printf("|%d,%lf,%lf|\t",mtx->colIdx_vertical[j],mtx->val_vertical[j*2],mtx->val_vertical[j*2+1]);
        }
        printf("\n");
    }
}
/** csr经过512行排序 **/
// 交换两个整数的值（通用交换函数，可用于原数组和索引数组）
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
// 分区函数：同时处理原数组和索引数组，保持位置映射同步
int partition(int arr[], int indexArr[], int left, int right) {
    // 选中间元素作为基准，避免有序数组下的最坏情况
    int mid = left + (right - left) / 2;
    int pivot = arr[mid];
    // 同时交换原数组和索引数组的基准元素，维持映射关系
    swap(&arr[left], &arr[mid]);
    swap(&indexArr[left], &indexArr[mid]);

    int i = left;  // 左指针（从基准值后一位开始）
    int j = right; // 右指针

    while (i < j) {
        // 从右向左找第一个小于基准值的元素
        while (i < j && arr[j] >= pivot) {
            j--;
        }
        // 从左向右找第一个大于基准值的元素
        while (i < j && arr[i] <= pivot) {
            i++;
        }
        // 同时交换原数组和索引数组的元素，保持位置对应
        if (i < j) {
            swap(&arr[i], &arr[j]);
            swap(&indexArr[i], &indexArr[j]);
        }
    }
    // 将基准值交换到最终位置，索引数组同步交换
    swap(&arr[left], &arr[i]);
    swap(&indexArr[left], &indexArr[i]);
    return i; // 返回基准值位置，用于分割子数组
}
void quickSort(int arr[], int indexArr[], int left, int right) {
    // 递归终止条件：子数组长度≤1
    if (left >= right) {
        return;
    }
    // 完成分区，获取基准值位置（原数组和索引数组同步分区）
    int pivotPos = partition(arr, indexArr, left, right);
    // 递归排序左子数组（基准值左侧）
    quickSort(arr, indexArr, left, pivotPos - 1);
    // 递归排序右子数组（基准值右侧）
    quickSort(arr, indexArr, pivotPos + 1, right);
}
void R_reorder(sparse_mtx *mat,sparse_mtx *re_mat,int *map_rfs,int window)
{
    re_mat->rownum = mat->rownum;
    re_mat->colnum = mat->colnum;
    re_mat->nnznum = mat->nnznum;
    re_mat->val = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE ) * mat->nnznum);
    re_mat->colidx = (int *) malloc(sizeof(int) * mat->nnznum);
    re_mat->rowptr = (int *) malloc(sizeof (int) * (mat->rownum + 1));
    re_mat->rowptr[0] = 0;

    int *record_num = (int *) malloc(sizeof(int)*window);
    int *indexArr = (int *) malloc(sizeof(int)*window);

    int b,i,j,z;
    b = (re_mat->rownum + (window - 1)) / window;

    int start_row,end_row;
    for(i = 0;i < b;i++) {
        start_row = i * window;
        if (i == b - 1) {
            end_row = re_mat->rownum;
        } else {
            end_row = (i + 1) * window;
        }
        //init
        for (j = start_row; j < end_row; j++) {
            record_num[j - start_row] = mat->rowptr[j + 1] - mat->rowptr[j];
            indexArr[j - start_row] = j - start_row;
        }
        //重排序
        quickSort(record_num, indexArr, 0, end_row - start_row - 1);
        //Rewrite
        for (j = start_row; j < end_row; j++) {
            map_rfs[j] = indexArr[end_row - j - 1] + start_row;
            re_mat->rowptr[j + 1] = re_mat->rowptr[j] + record_num[end_row - j - 1];
            for (z = re_mat->rowptr[j]; z < re_mat->rowptr[j + 1]; z++) {
                re_mat->colidx[z] = mat->colidx[mat->rowptr[start_row + indexArr[end_row - j - 1]] + z -
                                                      re_mat->rowptr[j]];
                re_mat->val[z] = mat->val[mat->rowptr[start_row + indexArr[end_row - j - 1]] + z -
                                          re_mat->rowptr[j]];
            }
        }
    }
    free(record_num);
    free(indexArr);
}
void trans_ell_c_a_mtx(sparse_mtx *mtx_csr,sparse_ell *mat_ESB,int window) {
    mat_ESB->rowNum = mtx_csr->rownum;
    mat_ESB->colNum = mtx_csr->colnum;
    mat_ESB->nnz = 0;
    int i, j, b, z;
    int start_row, end_row;
    b = mat_ESB->rowNum / window;

    mat_ESB->tilePtr = (int *) malloc(sizeof(int) * (b + 1));
    mat_ESB->tilePtr[0] = 0;
    mat_ESB->tile_num = b;
    for (i = 0; i < b; i++) {
        start_row = i * window;
        end_row = (i + 1) * window;
        mat_ESB->nnz += (mtx_csr->rowptr[i * window + 1] - mtx_csr->rowptr[i * window]) * (end_row - start_row);
        mat_ESB->tilePtr[i + 1] = mat_ESB->nnz;
    }
    mat_ESB->colIdx = (int *) malloc(sizeof(int) * mat_ESB->nnz);
    mat_ESB->val = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * mat_ESB->nnz);
    int count = 0;
    for (i = 0; i < b; i++) {
        start_row = i * window;
        end_row = (i + 1) * window;
        for (j = mtx_csr->rowptr[start_row]; j < mtx_csr->rowptr[start_row + 1]; j++) {
            for (z = start_row; z < end_row; z++) {
                if (j - mtx_csr->rowptr[i * window] < mtx_csr->rowptr[z + 1] - mtx_csr->rowptr[z]) {
                    mat_ESB->colIdx[count] = mtx_csr->colidx[mtx_csr->rowptr[z] + j - mtx_csr->rowptr[i * window]];
                    mat_ESB->val[count] = mtx_csr->val[mtx_csr->rowptr[z] + j - mtx_csr->rowptr[i * window]];
                } else {
                    mat_ESB->colIdx[count] = 0;
                    mat_ESB->val[count] = 0.0f;
                }
                count++;
            }
        }
    }
}
void Print_ellMtx_FP64(sparse_ell *mtx)
{
    printf("-----------------------row Ptr--------------------------\n");
    for(int i=0;i<mtx->rowNum/2;i++)
    {
        printf("row%d==%d,%d\n",i,mtx->tilePtr[i],mtx->tilePtr[i+1]);
    }
    printf("------------------------col-Idx--------------------------\n");
    for(int i=0;i<mtx->rowNum/2;i++)
    {
        for(int j=mtx->tilePtr[i];j<mtx->tilePtr[i+1];j+=2)
        {
            printf(" [%d %d] ",mtx->colIdx[j],mtx->colIdx[j+1]);
        }
        printf("\n");
    }
    printf("------------------------val--------------------------\n");
    for(int i=0;i<mtx->rowNum/2;i++)
    {
        for(int j=mtx->tilePtr[i];j<mtx->tilePtr[i+1];j+=2)
        {
            printf(" [%lf %lf] ",mtx->val[j],mtx->val[j+1]);
        }
        printf("\n");
    }
}
#endif