#ifndef OPERATOR_ACSR_NEON_H
#define OPERATOR_ACSR_NEON_H

#include "../struct/struct.h"
void spmv_cpu(sparse_mtx* mtx, VALUE_TYPE* vec, VALUE_TYPE* res) {
    VALUE_TYPE temp;
    for (int i = 0; i < mtx->rownum; i++) {
        temp = 0.0f;
        for (int j = mtx->rowptr[i]; j < mtx->rowptr[i + 1]; j++) {
            temp += mtx->val[j] * vec[mtx->colidx[j]];
        }
        res[i] = temp;
    }
}
void CSR_spmv_neon(sparse_mtx* mtx, VALUE_TYPE* x, VALUE_TYPE* y) {
    float64_t sum;
    float64x2_t tempy;
    float64x2_t matrix, vector;
    int i,j,end;
    for (i = 0; i < mtx->rownum; i++) {
        sum = 0.0;
        tempy = vdupq_n_f64(0.0);
        j = mtx->rowptr[i];
        end = mtx->rowptr[i + 1];
        for (; j <= end - 2; j += 2) {
            matrix = vld1q_f64(mtx->val + j);
            vector = vsetq_lane_f64(x[mtx->colidx[j]], vector, 0);
            vector = vsetq_lane_f64(x[mtx->colidx[j + 1]], vector, 1);
            tempy = vmlaq_f64(tempy, matrix, vector); //一直累加在向量寄存器中
        }
        sum += vgetq_lane_f64(tempy, 0) + vgetq_lane_f64(tempy, 1);
        for (; j < end; j++) {
            sum += mtx->val[j] * x[mtx->colidx[j]];
        }
        y[i] = sum;
    }
}
void ACSR_spmv_neon(VALUE_TYPE* result, acsr_mtx* mat, VALUE_TYPE* vec) {
    float64x2_t matVal, vecVal, mulRes;
    float64_t sum;
    for (int i = 0; i < mat->rownum; i++) {
        sum = 0.0;
        mulRes = vdupq_n_f64(0.0);
        for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++) {
            int index = j * VECTOR_LENGTH;
            matVal = vld1q_f64(mat->val + index);
            vecVal = vld1q_f64(vec + mat->colidx[j]);
            mulRes = vmlaq_f64(mulRes,matVal, vecVal);
        }
        sum += vgetq_lane_f64(mulRes, 0) + vgetq_lane_f64(mulRes, 1);
        result[i] = sum;
    }
}
void sy_SpMV(VALUE_TYPE *rec, symmetry_mtx *mtx,const VALUE_TYPE *b) {
    int k = mtx->rownum / 2;
    int i, j;
    float64x2_t mtxVal, recVal, bVal, temp, rec_row, diag, constbVal;
    float64_t row_1, row_2, per_row;
    float64_t r1,r2;
    for (i = 0; i < k; i++) {
        temp = vdupq_n_f64(0.0);
        constbVal = vld1q_f64(b + i * 2);
        r1 = b[i*2], r2 = b[i*2+1];
        //diag
        diag = vld1q_f64(mtx->val_diag + i * 2);//ld diag
        bVal = vld1q_f64(b + i * 2);//ld b
        rec_row = vmulq_f64(diag, bVal);//rec = diag * b
        //第一行水平
        for (j = mtx->rowPtr_horizontal[i * 2]; j < mtx->rowPtr_horizontal[i * 2 + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_horizontal + j * 2);//ld mtxVal
            bVal = vld1q_f64(b + mtx->colIdx_horizontal[j]);//ld b
            temp = vfmaq_f64(temp, mtxVal, bVal);//temp1+=mtx*b
            //associate
            recVal = vld1q_f64(rec + mtx->colIdx_horizontal[j]);//ld old rec
            recVal = vfmaq_n_f64(recVal,mtxVal,r1);
            vst1q_f64(rec + mtx->colIdx_horizontal[j], recVal);//store rec
        }
        row_1 = vaddvq_f64(temp);
        temp = vdupq_n_f64(0.0);
        //第二行水平
        for (j = mtx->rowPtr_horizontal[i * 2 + 1]; j < mtx->rowPtr_horizontal[(i + 1) * 2]; j++) {
            mtxVal = vld1q_f64(mtx->val_horizontal + j * 2);
            bVal = vld1q_f64(b + mtx->colIdx_horizontal[j]);
            temp = vfmaq_f64(temp, mtxVal, bVal);//temp1+=mtx*b
            //associate
            recVal = vld1q_f64(rec + mtx->colIdx_horizontal[j]);//Get rec
            recVal = vfmaq_n_f64(recVal,mtxVal,r2);
            vst1q_f64(rec + mtx->colIdx_horizontal[j], recVal);//store rec
        }
        row_2 = vaddvq_f64(temp);
        rec_row = vaddq_f64(vcombine_f64(vmov_n_f64(row_1), vmov_n_f64(row_2)), rec_row);//完成水平计算
        //斜角
        for (j = mtx->rowPtr_bevel[i]; j < mtx->rowPtr_bevel[i + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_bevel + j * 2);//ld mtx
            bVal = vld1q_f64(b + mtx->colIdx_bevel[j]);//ld b
            rec_row = vfmaq_f64(rec_row,mtxVal,bVal);
            //associate
            recVal = vld1q_f64(rec + mtx->colIdx_bevel[j]);//ld recVal
            recVal = vfmaq_f64(recVal,constbVal,mtxVal);
            vst1q_f64(rec + mtx->colIdx_bevel[j], recVal);//st
        }
        //垂直
        for (j = mtx->rowPtr_vertical[i]; j < mtx->rowPtr_vertical[i + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_vertical + j * 2);//ld mtx
            rec_row = vfmaq_n_f64(rec_row,mtxVal,b[mtx->colIdx_vertical[j]]);
            //associate
            per_row = vaddvq_f64(vmulq_f64(mtxVal, constbVal));
            rec[mtx->colIdx_vertical[j]] += per_row;
        }
        rec_row = vaddq_f64(vld1q_f64(rec + i * 2), rec_row);
        vst1q_f64(rec + i * 2, rec_row);
    }
}
void asy_SpMV(VALUE_TYPE *rec, symmetry_mtx *mtx,const VALUE_TYPE *b) {
    int k = mtx->rownum / 2;
    int i, j;
    float64x2_t mtxVal, bVal, temp, rec_row;
    float64_t row_1, row_2;
    for (i = 0; i < k; i++) {
        temp = vdupq_n_f64(0.0);
        //第一行水平
        for (j = mtx->rowPtr_horizontal[i * 2]; j < mtx->rowPtr_horizontal[i * 2 + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_horizontal + j * 2);//ld mtxVal
            bVal = vld1q_f64(b + mtx->colIdx_horizontal[j]);//ld b
            temp = vfmaq_f64(temp, mtxVal, bVal);//temp1+=mtx*b
        }
        row_1 = vaddvq_f64(temp);
        temp = vdupq_n_f64(0.0);
        //第二行水平
        for (j = mtx->rowPtr_horizontal[i * 2 + 1]; j < mtx->rowPtr_horizontal[(i + 1) * 2]; j++) {
            mtxVal = vld1q_f64(mtx->val_horizontal + j * 2);
            bVal = vld1q_f64(b + mtx->colIdx_horizontal[j]);
            temp = vfmaq_f64(temp, mtxVal, bVal);//temp1+=mtx*b
        }
        row_2 = vaddvq_f64(temp);
        rec_row = vcombine_f64(vmov_n_f64(row_1), vmov_n_f64(row_2));//完成水平计算
        //斜角
        for (j = mtx->rowPtr_bevel[i]; j < mtx->rowPtr_bevel[i + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_bevel + j * 2);//ld mtx
            bVal = vld1q_f64(b + mtx->colIdx_bevel[j]);//ld b
            rec_row = vfmaq_f64(rec_row,mtxVal,bVal);
        }
        //垂直
        for (j = mtx->rowPtr_vertical[i]; j < mtx->rowPtr_vertical[i + 1]; j++) {
            mtxVal = vld1q_f64(mtx->val_vertical + j * 2);//ld mtx
            rec_row = vfmaq_n_f64(rec_row,mtxVal,b[mtx->colIdx_vertical[j]]);
        }
        vst1q_f64(rec + i * 2, rec_row);
    }
}
void ELL_spmv_neon_FP64(sparse_ell *mtx_ell,VALUE_TYPE *vec_b,VALUE_TYPE *temp_ell) {
    float64x2_t temp, mtxVal, vec;
    for (int i = 0; i < mtx_ell->rowNum / 2; i++) {
        temp = vdupq_n_f64(0.0f);
        for (int j = mtx_ell->tilePtr[i]; j < mtx_ell->tilePtr[i + 1]; j += 2) {
            vec = vsetq_lane_f64(vec_b[mtx_ell->colIdx[j]], vec, 0);
            vec = vsetq_lane_f64(vec_b[mtx_ell->colIdx[j + 1]], vec, 1);
            mtxVal = vld1q_f64(mtx_ell->val + j);
            temp = vmlaq_f64(temp,mtxVal, vec);
        }
        vst1q_f64(temp_ell + i * 2, temp);
    }
}
#endif
