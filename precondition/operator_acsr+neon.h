//
// Created by lenovo on 2023/9/15.
//

#ifndef OPERATOR_ACSR_NEON_H
#define OPERATOR_ACSR_NEON_H
//
// Created by lenovo on 2023/9/15.
//
#include "../../include/common.h"
void ACSR_spmv_neon(VALUE_TYPE* result, acsr_mtx* mat, VALUE_TYPE* vec)
{
    for (int i = 0; i < mat->rownum; i++)
    {
#if VALUE_TYPE_WIDTH == 32
        float32_t sum = 0.0;
        float32x4_t matVal, vecVal, mulRes;
            for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++)
            {
                int index = j * VECTOR_LENGTH;
                matVal = vld1q_f32(mat->val + index);
                vecVal = vld1q_f32(vec + mat->colidx[j]);
                mulRes = vmulq_f32(matVal, vecVal);
                sum += vaddvq_f32(mulRes);
            }
#elif VALUE_TYPE_WIDTH == 64
        float64_t sum = 0.0;
        float64x2_t matVal, vecVal, mulRes;
        for (int j = mat->rowptr[i]; j < mat->rowptr[i + 1]; j++)
        {
            int index = j * VECTOR_LENGTH;
            matVal = vld1q_f64(mat->val + index);
            vecVal = vld1q_f64(vec + mat->colidx[j]);
            mulRes = vmulq_f64(matVal, vecVal);
            sum += vaddvq_f64(mulRes);
        }
#endif
        result[i] = sum;
    }
}

void vMulv_neon(VALUE_TYPE* res, VALUE_TYPE* vec1, VALUE_TYPE* vec2, int n)
{
    *res = 0.0;
    const VALUE_TYPE* vec1_t = vec1;
    const VALUE_TYPE* vec2_t = vec2;
    int i;
#if VALUE_TYPE_WIDTH == 32
    float32_t sum_32=0.0;
    float32x4_t in1, in2, out;
    for (i = 0; i < n - 4; i += 4)
    {
        //128位寄存器
        in1 = vld1q_f32(vec1_t+i);
        in2 = vld1q_f32(vec2_t+i);
        out = vmulq_f32(in1, in2);
        sum_32 += vaddvq_f32(out);
    }
    for (; i < n; i++)
    {
        //直接内存加载相乘
        sum_32 += *(vec1_t+i) * *(vec2_t+i);
    }
    *res=sum_32;
#elif VALUE_TYPE_WIDTH == 64
    float64_t sum_64=0.0;
    float64x2_t in1, in2, out;
    for (i = 0; i < n - 2; i += 2)
    {
        //128位寄存器
        in1 = vld1q_f64(vec1_t+i);
        in2 = vld1q_f64(vec2_t+i);
        out = vmulq_f64(in1, in2);
        sum_64 += vaddvq_f64(out);//自动规约
    }
    for (; i < n; i++)
    {
        //直接内存加载相乘
        sum_64 += *(vec1_t+i) * *(vec2_t+i);
    }
    *res=sum_64;
#endif

}
void vAddvMuls_neon(VALUE_TYPE *res, VALUE_TYPE *vec1, VALUE_TYPE *vec2, VALUE_TYPE sca, int n)
{
    VALUE_TYPE *res_t = res;
    const VALUE_TYPE *vec1_t = vec1;
    const VALUE_TYPE *vec2_t = vec2;
    int i;
#if VALUE_TYPE_WIDTH == 32
    float32x4_t in1, in2, in3, out;
    float32_t scalar = sca;
        for (i = 0; i < n - 4; i += 4)
        {
            in1 = vld1q_f32(vec1_t + i);
            in2 = vld1q_f32(vec2_t + i);
            in3 = vmulq_n_f32(in2, scalar);
            out = vaddq_f32(in1, in3);
            vst1q_f32(res_t+i, out);
        }
        for (; i < n; i++)
        {
            *(res_t+i) = *(vec1_t+i) + *(vec2_t+i) * sca;
        }
#elif VALUE_TYPE_WIDTH == 64
    float64x2_t in1, in2, in3, out;
    float64_t scalar = sca;
    for (i = 0; i < n - 2; i += 2)
    {
        in1 = vld1q_f64(vec1_t + i);
        in2 = vld1q_f64(vec2_t + i);
        in3 = vmulq_n_f64(in2, scalar);
        out = vaddq_f64(in1, in3);
        vst1q_f64(res_t+i, out);
    }
    for (; i < n; i++)
    {
        *(res_t+i) = *(vec1_t+i) + *(vec2_t+i) * sca;
    }
#endif
}
VALUE_TYPE vEuklid_neon(VALUE_TYPE* src, int n)
{
    VALUE_TYPE ans = 0.0;
    vMulv_neon(&ans, src, src, n);
    return sqrt(ans);
}


#endif //PCG_OPERATOR_ACSR_NEON_H
