#ifndef _UTILS_
#define _UTILS_

#include "../../include/common.h"
#include "arm_neon.h"

void binary_search_right_boundary_item_kernel(const MAT_PTR_TYPE* row_pointer, const MAT_PTR_TYPE key_input, const int size, 
                                              int* colpos, MAT_PTR_TYPE* nnzpos)
{
    int start = 0;
    int stop = size - 1;
    MAT_PTR_TYPE median;
    MAT_PTR_TYPE key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
        {
            start = median + 1;
        }
        else
        {
            stop = median - 1;
        }
    }

    *colpos = start - 1;
    *nnzpos = key_input - row_pointer[*colpos];
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

void swap_key(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_val(MAT_VAL_TYPE* a , MAT_VAL_TYPE* b)
{
    MAT_VAL_TYPE tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
int partition_key_val_pair(int* key, MAT_VAL_TYPE* val, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_val(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            swap_val(&val[pivot_index+i], &val[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_val(&val[pivot_index + length - 1], &val[small_length]);

    return small_length;
}

// quick sort key-value pair (main function)
void quick_sort_key_val_pair(int* key, MAT_VAL_TYPE* val, int length)
{
    if(length == 0 || length == 1) return;

    int small_length = partition_key_val_pair(key, val, length, 0);
    quick_sort_key_val_pair(key, val, small_length);
    quick_sort_key_val_pair(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}

// quick sort key (child function)
int partition_key(int* key, int length, int pivot_index)
{
    int i = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key(&key[pivot_index+i], &key[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);

    return small_length;
}

// quick sort key (main function)
void quick_sort_key(int* key, int length)
{
    if(length == 0 || length == 1) return;

    int small_length = partition_key(key, length, 0);
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}

void matrix_transposition(const int m, const int n, const MAT_PTR_TYPE nnz, const MAT_PTR_TYPE* csrRowPtr, const int* csrColIdx,
                          const MAT_VAL_TYPE* csrVal, int* cscRowIdx, MAT_PTR_TYPE* cscColPtr, MAT_VAL_TYPE* cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(MAT_PTR_TYPE) * (n+1));
    for (MAT_PTR_TYPE i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    MAT_PTR_TYPE *cscColIncr = (MAT_PTR_TYPE*)malloc(sizeof(MAT_PTR_TYPE) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(MAT_PTR_TYPE) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (MAT_PTR_TYPE j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
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


void* spmv_serial(sparse_mtx* mtx, VALUE_TYPE* vec, VALUE_TYPE* res)
{
    for (int i = 0; i < mtx->rownum; i++)
    {
        res[i] = 0;
        for (int j = mtx->rowptr[i]; j < mtx->rowptr[i+1]; j++)
        {
            res[i] += mtx->val[j] * vec[mtx->colidx[j]];
        }
    }
    return NULL;
}


void* pthread_process (void* arg)
{
    thread* pn = (thread*)arg;
    for (int i = pn->start; i < pn->end; i++)
    {
        VALUE_TYPE sum = 0;
        for (int j = pn->rowptr[i]; j < pn->rowptr[i + 1]; j++)
        {
            sum += pn->value[j] * pn->X[pn->colidx[j]];
        }
        pn->Y[i] = sum;
    }
    return NULL;
}

void* neon_process(void* arg)
{
    thread* pn = (thread*)arg;
    float32x4_t left, right, total;
    for (int i = pn->start; i < pn->end; i++)
    {
        VALUE_TYPE sum = 0;
        for (int j = pn->rowptr[i]; j < pn->rowptr[i+1]; j++)
        {
            int index = j * VECTOR_LENGTH;
            left = vld1q_f32(reinterpret_cast<const float32_t *>(pn->value + index));
            right = vld1q_f32(reinterpret_cast<const float32_t *>(pn->X + pn->colidx[j]));
            total = vmulq_f32(left, right);
            sum += vaddvq_f32(total);
        }
        pn->Y[i] = sum;
    }
    return NULL;
}

void freemtx(sparse_mtx* mtx)
{
    free(mtx->rowptr);
    free(mtx->colidx);
    free(mtx->val);
}

int check(int n, VALUE_TYPE* val1, VALUE_TYPE* val2)
{
    int error = 0;
    for (int i = 0; i < n; i++)
    {
        if (val1[i] != val2[i])
        {
            error++;
        }
    }
    return error;
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
acsr_mtx * tran_ACSR(sparse_mtx mtx)
{
    acsr_mtx *temp= (acsr_mtx*)malloc(sizeof(acsr_mtx));
    temp->rownum=mtx.rownum;//行数复制
    temp->colnum=mtx.colnum;//列数复制
    temp->nnznum=mtx.nnznum;//非零元复制
    temp->vecnum=0;//向量对个数
    //temp.colidx;//列索引
    temp->rowptr=(int *)malloc(sizeof (int)*mtx.rownum);//分配多少行就多少个列索引
    //temp.val;//值索引
    int start_id,end_id;
    VALUE_TYPE data[2];
    /**
     * 首先统计vector个数,再根据个数分配内存
     */
    for(int row=0;row<mtx.rownum;row++)
    {
        start_id=mtx.rowptr[row],end_id=mtx.rowptr[row+1];
        temp->rowptr[row]=temp->vecnum;
        for(int i=start_id;i<end_id;i++)
        {
//            data[0]=mtx.val[i];
//            temp.colidx[colidx++]=mtx.colidx[i];
            if(((i+1)!=end_id)&&mtx.colidx[i+1]-mtx.colidx[i]==1)
            {
//                data[1]=mtx.val[++i];
                ++i;
            }
            else{
                data[1]=0;//填入零元素
            }
            temp->vecnum++;
        }
    }
    temp->colidx=(int *) malloc(sizeof (int)*temp->vecnum);
    int colidx=0;
    temp->val=(VALUE_TYPE *) malloc(sizeof (VALUE_TYPE)*temp->vecnum*2);
    int val=0;
    temp->vecnum=0;
    for(int row=0;row<mtx.rownum;row++)
    {
        start_id=mtx.rowptr[row],end_id=mtx.rowptr[row+1];
        //cout<<start_id<<","<<end_id<<endl;
        temp->rowptr[row]=temp->vecnum;
        for(int i=start_id;i<end_id;i++)
        {
            temp->val[val++]=mtx.val[i];
            temp->colidx[colidx++]=mtx.colidx[i];
            if(((i+1)!=end_id)&&mtx.colidx[i+1]-mtx.colidx[i]==1)
            {
                temp->val[val++]=mtx.val[++i];
            }
            else{
                temp->val[val++]=0;//填入零元素
            }
            temp->vecnum++;
        }
    }
    return temp;
}

#endif