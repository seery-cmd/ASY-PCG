//
// Created by lenovo on 2026/9/16.
//

#ifndef NEON_SPMV_SPMV_KERNEL_FP32_SY_H
#define NEON_SPMV_SPMV_KERNEL_FP32_SY_H

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
    for (int i = 0; i < mtx->rownum; i++) {
        float32_t sum = 0.0;
        float32x4_t tempy = vdupq_n_f32(0.0);
        int j = mtx->rowptr[i];
        int end = mtx->rowptr[i + 1];
        float32x4_t matrix, vector;
        for (; j <= end - 4; j += 4) {
            matrix = vld1q_f32(mtx->val + j);
            vector = vsetq_lane_f32(x[mtx->colidx[j]], vector, 0);
            vector = vsetq_lane_f32(x[mtx->colidx[j + 1]], vector, 1);
            vector = vsetq_lane_f32(x[mtx->colidx[j + 2]], vector, 2);
            vector = vsetq_lane_f32(x[mtx->colidx[j + 3]], vector, 3);
            tempy = vmlaq_f32(tempy, matrix, vector); //一直累加在向量寄存器中
        }
        sum += vgetq_lane_f32(tempy, 0) + vgetq_lane_f32(tempy, 1) + vgetq_lane_f32(tempy, 2) + vgetq_lane_f32(tempy, 3);
        for (; j < end; j++) {
            sum += mtx->val[j] * x[mtx->colidx[j]];
        }
        y[i] = sum;
    }
}
void ELL_spmv_neon_FP32(sparse_ell *mtx_ell,VALUE_TYPE *vec_b,VALUE_TYPE *temp_ell) {
    float32x4_t temp, mtxVal, vec;
    for (int i = 0; i < mtx_ell->rowNum / 4; i++) {
        temp = vdupq_n_f32(0.0f);
        for (int j = mtx_ell->tilePtr[i]; j < mtx_ell->tilePtr[i + 1]; j += 4) {
            vec = vsetq_lane_f32(vec_b[mtx_ell->colIdx[j]], vec, 0);
            vec = vsetq_lane_f32(vec_b[mtx_ell->colIdx[j + 1]], vec, 1);
            vec = vsetq_lane_f32(vec_b[mtx_ell->colIdx[j + 2]], vec, 2);
            vec = vsetq_lane_f32(vec_b[mtx_ell->colIdx[j + 3]], vec, 3);
            mtxVal = vld1q_f32(mtx_ell->val + j);
            temp = vaddq_f32(vmulq_f32(vec, mtxVal), temp);
        }
        vst1q_f32(temp_ell + i * 4, temp);
    }
}
void ACSR_spmv_neon(VALUE_TYPE* result, acsr_mtx* mat, VALUE_TYPE* vec) {
    float32_t sum ;
    float32x4_t matVal, vecVal, mulRes;
    for (int i = 0; i < mat->rownum; i++) {
        sum = 0.0;
        mulRes = vdupq_n_f32(0.0);
        for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++) {
            int index = j * VECTOR_LENGTH;
            matVal = vld1q_f32(mat->val + index);
            vecVal = vld1q_f32(vec + mat->colidx[j]);
            mulRes = vmlaq_f32(mulRes,matVal, vecVal);
        }
        sum += vgetq_lane_f32(mulRes, 0) + vgetq_lane_f32(mulRes, 1) + vgetq_lane_f32(mulRes, 2) + vgetq_lane_f32(mulRes, 3);
        result[i] = sum;
    }
}
void sy_SpMV(VALUE_TYPE *rec, symmetry_mtx *mtx,const VALUE_TYPE *b) {
    int k = mtx->rownum / 4;
    int i, j;
    float32x4_t mtxVal, bVal, recVal, diag, constbVal[2],temp,recAs;
    float32x2_t A,B,R,recas,constbval[2],f[2];
    float32_t row_1[2], row_2[2], per_row;
    float32_t r1[2],r2[2];
    for (i = 0; i < k; i++) {
        //f表示斜角和垂直的最终值,row表示水平的最终值
        r1[0] = b[i*4], r2[0] = b[i*4+1];
        r1[1] = b[i*4+2], r2[1] = b[i*4+3];
        constbVal[0] = vcombine_f32(vld1_f32(b + i*4),vld1_f32(b + i*4)),constbVal[1] = vcombine_f32(vld1_f32(b + i*4 + 2),vld1_f32(b + i*4 + 2));
        constbval[0] = vld1_f32(b + i*4),constbval[1] = vld1_f32(b + i*4 + 2);
        f[0] = vdup_n_f32(0.0),f[1] = vdup_n_f32(0.0);
        //diag
        diag = vld1q_f32(mtx->val_diag + i * 4);//ld diag
        bVal = vld1q_f32(b + i * 4);//ld b
        recVal = vmulq_f32(diag, bVal);//rec = diag * b
        for (int z = 0; z < 2; z++) {
            //first row
            temp = vdupq_n_f32(0.0);//32*4
            R = vdup_n_f32(0.0);//32*2
            for (j = mtx->rowPtr_horizontal[i * 4 + z*2]; j < mtx->rowPtr_horizontal[i * 4 + 1 + z*2] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_horizontal + j * 2);//ld mtxVal
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_horizontal[j]),vld1_f32(b + mtx->colIdx_horizontal[j+1]));//ld b
                temp = vfmaq_f32(temp, mtxVal, bVal);//scalar temp1 += mtx*b
                //associate
                recAs = vcombine_f32(vld1_f32(rec + mtx->colIdx_horizontal[j]),vld1_f32(rec + mtx->colIdx_horizontal[j+1]));//ld old rec
                recAs = vfmaq_n_f32(recAs,mtxVal,r1[z]);//new_rec = old_rec + temp
                vst1_f32(rec + mtx->colIdx_horizontal[j], vget_low_f32(recAs));
                vst1_f32(rec + mtx->colIdx_horizontal[j+1], vget_high_f32(recAs));
            }
            row_1[z] = vaddvq_f32(temp);
            //leave first row
            for(;j < mtx->rowPtr_horizontal[i * 4 + 1 + z*2];j++)
            {
                A = vld1_f32(mtx->val_horizontal + j * 2);
                B = vld1_f32(b + mtx->colIdx_horizontal[j]);//ld b
                R = vfma_f32(R, B, A);//scalar temp1 += mtx*b
                //associate
                recas = vld1_f32(rec + mtx->colIdx_horizontal[j]);
                recas = vmla_n_f32(recas,A,r1[z]);
                vst1_f32(rec + mtx->colIdx_horizontal[j],recas);
            }
            row_1[z] += vaddv_f32(R);
            //second row
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_horizontal[i * 4 + 1 + z*2]; j < mtx->rowPtr_horizontal[i * 4 + z*2 + 2] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_horizontal + j * 2);//ld mtxVal
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_horizontal[j]),vld1_f32(b + mtx->colIdx_horizontal[j+1]));//ld b
                temp = vfmaq_f32(temp, mtxVal, bVal);//scalar temp1 += mtx*b
                //associate
                recAs = vcombine_f32(vld1_f32(rec + mtx->colIdx_horizontal[j]),vld1_f32(rec + mtx->colIdx_horizontal[j+1]));//ld old rec
                recAs = vfmaq_n_f32(recAs,mtxVal,r2[z]);//new_rec = old_rec + temp
                vst1_f32(rec + mtx->colIdx_horizontal[j], vget_low_f32(recAs));
                vst1_f32(rec + mtx->colIdx_horizontal[j+1], vget_high_f32(recAs));
            }
            row_2[z] = vaddvq_f32(temp);
            //leave second row
            for(;j < mtx->rowPtr_horizontal[i * 4 + z*2 + 2];j++)
            {
                A = vld1_f32(mtx->val_horizontal + j * 2);
                B = vld1_f32(b + mtx->colIdx_horizontal[j]);//ld b
                R = vfma_f32(R, B, A);//scalar temp1 += mtx*b
                //associate
                recas = vld1_f32(rec + mtx->colIdx_horizontal[j]);
                recas = vmla_n_f32(recas,A,r2[z]);
                vst1_f32(rec + mtx->colIdx_horizontal[j],recas);
            }
            row_2[z] += vaddv_f32(R);
            //斜角
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_bevel[i*2 + z]; j < mtx->rowPtr_bevel[i*2 + 1 + z] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_bevel + j * 2);//ld mtx
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_bevel[j]),vld1_f32(b + mtx->colIdx_bevel[j+1]));//ld b
                temp = vfmaq_f32(temp,mtxVal,bVal);//temp += mtx * b
                //associate
                recAs = vcombine_f32(vld1_f32(rec + mtx->colIdx_bevel[j]),vld1_f32(rec + mtx->colIdx_bevel[j+1]));//ld recVal
                recAs = vfmaq_f32(recAs,mtxVal,constbVal[z]);
                vst1_f32(rec + mtx->colIdx_bevel[j], vget_low_f32(recAs));
                vst1_f32(rec + mtx->colIdx_bevel[j+1], vget_high_f32(recAs));
            }
            for(;j<mtx->rowPtr_bevel[i*2 + 1 + z];j++)
            {
                A = vld1_f32(mtx->val_bevel + j * 2);
                B = vld1_f32(b + mtx->colIdx_bevel[j]);
                R = vfma_f32(R, A, B);
                //associate
                recas = vld1_f32(rec + mtx->colIdx_bevel[j]);
                recas = vfma_f32(recas,A,constbval[z]);
                vst1_f32(rec + mtx->colIdx_bevel[j],recas);
            }
            recas = vadd_f32(vget_low_f32(temp),vget_high_f32(temp));
            f[z] = vadd_f32(recas,R);
            //垂直
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_vertical[i*2 + z]; j < mtx->rowPtr_vertical[i*2 + 1 + z]-2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_vertical + j * 2);//ld mtx
                bVal = vcombine_f32(vdup_n_f32(b[mtx->colIdx_vertical[j]]),vdup_n_f32(b[mtx->colIdx_vertical[j+1]]));
                temp = vfmaq_f32(temp,mtxVal,bVal);
                //associate
                recAs = vmulq_f32(mtxVal,constbVal[z]);
                per_row = vaddv_f32(vget_low_f32(recAs));
                rec[mtx->colIdx_vertical[j]] += per_row;
                per_row = vaddv_f32(vget_high_f32(recAs));
                rec[mtx->colIdx_vertical[j+1]] += per_row;
            }
            for(;j<mtx->rowPtr_vertical[i*2 + 1 + z];j++) {
                A = vld1_f32(mtx->val_vertical + j * 2);
                B = vdup_n_f32(b[mtx->colIdx_vertical[j]]);
                R = vfma_f32(R, A, B);
                //associate
                recas = vmul_f32(A,constbval[z]);
                per_row = vaddv_f32(recas);
                rec[mtx->colIdx_vertical[j]] += per_row;
            }
            recas = vadd_f32(vget_low_f32(temp),vget_high_f32(temp));
            recas = vadd_f32(recas,R);
            f[z] = vadd_f32(f[z],recas);
        }
        recAs = vcombine_f32(f[0],f[1]);
        recVal = vaddq_f32(recVal,recAs);

        recAs = vsetq_lane_f32(row_1[0], recAs, 0);
        recAs = vsetq_lane_f32(row_2[0], recAs, 1);
        recAs = vsetq_lane_f32(row_1[1], recAs, 2);
        recAs = vsetq_lane_f32(row_2[1], recAs, 3);
        recVal = vaddq_f32(recVal,recAs);

        recVal = vaddq_f32(vld1q_f32(rec + i * 4), recVal);
        vst1q_f32(rec + i * 4, recVal);
    }
}
void asy_SpMV(VALUE_TYPE *rec, symmetry_mtx *mtx,const VALUE_TYPE *b) {
    int k = mtx->rownum / 4;
    int i, j;
    float32x4_t mtxVal, bVal, recVal,temp,recAs;
    float32x2_t A,B,R,recas,f[2];
    float32_t row_1[2], row_2[2], per_row;
    for (i = 0; i < k; i++) {
        //f表示斜角和垂直的最终值,row表示水平的最终值
        f[0] = vdup_n_f32(0.0),f[1] = vdup_n_f32(0.0);
        for (int z = 0; z < 2; z++) {
            //first row
            temp = vdupq_n_f32(0.0);//32*4
            R = vdup_n_f32(0.0);//32*2
            for (j = mtx->rowPtr_horizontal[i * 4 + z*2]; j < mtx->rowPtr_horizontal[i * 4 + 1 + z*2] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_horizontal + j * 2);//ld mtxVal
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_horizontal[j]),vld1_f32(b + mtx->colIdx_horizontal[j+1]));//ld b
                temp = vfmaq_f32(temp, mtxVal, bVal);//scalar temp1 += mtx*b
            }
            row_1[z] = vaddvq_f32(temp);
            //leave first row
            for(;j < mtx->rowPtr_horizontal[i * 4 + 1 + z*2];j++)
            {
                A = vld1_f32(mtx->val_horizontal + j * 2);
                B = vld1_f32(b + mtx->colIdx_horizontal[j]);//ld b
                R = vfma_f32(R, B, A);//scalar temp1 += mtx*b
            }
            row_1[z] += vaddv_f32(R);
            //second row
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_horizontal[i * 4 + 1 + z*2]; j < mtx->rowPtr_horizontal[i * 4 + z*2 + 2] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_horizontal + j * 2);//ld mtxVal
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_horizontal[j]),vld1_f32(b + mtx->colIdx_horizontal[j+1]));//ld b
                temp = vfmaq_f32(temp, mtxVal, bVal);//scalar temp1 += mtx*b
            }
            row_2[z] = vaddvq_f32(temp);
            //leave second row
            for(;j < mtx->rowPtr_horizontal[i * 4 + z*2 + 2];j++)
            {
                A = vld1_f32(mtx->val_horizontal + j * 2);
                B = vld1_f32(b + mtx->colIdx_horizontal[j]);//ld b
                R = vfma_f32(R, B, A);//scalar temp1 += mtx*b
            }
            row_2[z] += vaddv_f32(R);
            //斜角
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_bevel[i*2 + z]; j < mtx->rowPtr_bevel[i*2 + 1 + z] - 2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_bevel + j * 2);//ld mtx
                bVal = vcombine_f32(vld1_f32(b + mtx->colIdx_bevel[j]),vld1_f32(b + mtx->colIdx_bevel[j+1]));//ld b
                temp = vfmaq_f32(temp,mtxVal,bVal);//temp += mtx * b
            }
            for(;j<mtx->rowPtr_bevel[i*2 + 1 + z];j++)
            {
                A = vld1_f32(mtx->val_bevel + j * 2);
                B = vld1_f32(b + mtx->colIdx_bevel[j]);
                R = vfma_f32(R, A, B);
            }
            recas = vadd_f32(vget_low_f32(temp),vget_high_f32(temp));
            f[z] = vadd_f32(recas,R);
            //垂直
            temp = vdupq_n_f32(0.0);
            R = vdup_n_f32(0.0);
            for (j = mtx->rowPtr_vertical[i*2 + z]; j < mtx->rowPtr_vertical[i*2 + 1 + z]-2; j+=2) {
                mtxVal = vld1q_f32(mtx->val_vertical + j * 2);//ld mtx
                bVal = vcombine_f32(vdup_n_f32(b[mtx->colIdx_vertical[j]]),vdup_n_f32(b[mtx->colIdx_vertical[j+1]]));
                temp = vfmaq_f32(temp,mtxVal,bVal);
            }
            for(;j<mtx->rowPtr_vertical[i*2 + 1 + z];j++) {
                A = vld1_f32(mtx->val_vertical + j * 2);
                B = vdup_n_f32(b[mtx->colIdx_vertical[j]]);
                R = vfma_f32(R, A, B);
            }
            recas = vadd_f32(vget_low_f32(temp),vget_high_f32(temp));
            recas = vadd_f32(recas,R);
            f[z] = vadd_f32(f[z],recas);
        }
        recVal = vcombine_f32(f[0],f[1]);

        recAs = vsetq_lane_f32(row_1[0], recAs, 0);
        recAs = vsetq_lane_f32(row_2[0], recAs, 1);
        recAs = vsetq_lane_f32(row_1[1], recAs, 2);
        recAs = vsetq_lane_f32(row_2[1], recAs, 3);
        recVal = vaddq_f32(recVal,recAs);

        vst1q_f32(rec + i * 4, recVal);
    }
}
#endif //NEON_SPMV_SPMV_KERNEL_FP32_SY_H
