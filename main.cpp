#include "src/struct/struct.h"
#include "src/kernel/spmv_kernel_FP32_sy.h"
#include "src/readMtx/mmio_highlevel.h"
#include <iostream>
#include <fstream>
#include <kspblas.h>
int main(int argc, char* argv[]) {
    int epoch = 10;
    int is_sy = 0;
    sparse_mtx *mtx_csr=(sparse_mtx *)malloc(sizeof(sparse_mtx));
    readmtx(argv[1], mtx_csr,is_sy);
    if(mtx_csr->rownum < 3000 || mtx_csr->rownum > 5000000) {
        printf("matrix is too small or big\n");
        return 0;
    }
    printf("mtx:%s,%d,%d,%d\n",argv[1],mtx_csr->rownum,mtx_csr->colnum,mtx_csr->nnznum);
    int error_num;
    VALUE_TYPE *vec_b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*(mtx_csr->colnum + 1));
    VALUE_TYPE *rec_cpu = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx_csr->rownum);
    VALUE_TYPE *rec_csr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mtx_csr->rownum);

    VALUE_TYPE *rec_sell_c_a = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mtx_csr->rownum);

    VALUE_TYPE *rec_acsr = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx_csr->rownum);
    VALUE_TYPE *rec_sy = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE )*(mtx_csr->rownum + 1));
    VALUE_TYPE *rec_kml = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx_csr->rownum);
    vec_b[0] = 1.0f;
    for(int i=0;i<mtx_csr->colnum;i++)
    {
        vec_b[i + 1] = 1.0f;
    }
    rec_sy[0] = 0.0f;
    for(int i=0;i<mtx_csr->rownum;i++)
    {
        rec_cpu[i] = 0.0f;
        rec_csr[i] = 0.0f;
        rec_sell_c_a[i] = 0.0f; // 初始化
        rec_acsr[i] = 0.0f;
        rec_sy[i + 1] = 0.0f;
    }
    for(int i=0;i<mtx_csr->nnznum;i++)
    {
        mtx_csr->val[i] = 1.0f; //对称矩阵时,得保证对称,不然map操作会出错
    }
    struct timeval t1_cpu, t2_cpu;
    struct timeval t1_neon, t2_neon;
/** CPU **/
    gettimeofday(&t1_cpu, NULL);
    for(int i=0;i<epoch;i++) {
        spmv_cpu(mtx_csr, vec_b, rec_cpu);
    }
    gettimeofday(&t2_cpu, NULL);
    VALUE_TYPE time_cpu = (t2_cpu.tv_sec - t1_cpu.tv_sec) * 1000.0 + (t2_cpu.tv_usec - t1_cpu.tv_usec) / 1000.0;
    printf("cpu_CSR time cost %lf ms\n",time_cpu/epoch);
/** CSR **/
    gettimeofday(&t1_neon, NULL);
    for(int i=0;i<epoch;i++) {
        CSR_spmv_neon(mtx_csr, vec_b, rec_csr);
    }
    gettimeofday(&t2_neon, NULL);
    VALUE_TYPE time_csr = (t2_neon.tv_sec - t1_neon.tv_sec) * 1000.0 + (t2_neon.tv_usec - t1_neon.tv_usec) / 1000.0;
    printf("neon_CSR time cost %lf ms\n", time_csr/epoch);
    error_num = 0;
    for(int i=0; i<mtx_csr->rownum; i++) {
        if(fabs(rec_cpu[i] - rec_csr[i]) > 1e-5) {
            if (error_num == 0) {
                printf("error! tid=%d %lf,%lf\n", i, rec_cpu[i], rec_sell_c_a[i]);
            }
            error_num++;
        }
    }
    printf("CSR error:%d\n",error_num);
///** sell-c-a && compare**/
    int *map_512 = (int *) malloc(sizeof(int)*mtx_csr->colnum);
    VALUE_TYPE *temp_ell = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx_csr->colnum);
    sparse_mtx *mtx512_csr = (sparse_mtx *)malloc(sizeof(sparse_mtx));
    R_reorder(mtx_csr,mtx512_csr,map_512,512);
    //Print_csr(sort_csr);
    sparse_ell *mtx_esb = (sparse_ell *) malloc(sizeof(sparse_ell));
    trans_ell_c_a_mtx(mtx512_csr,mtx_esb,VECTOR_LENGTH);

    gettimeofday(&t1_neon, NULL);
    for(int i=0;i<epoch;i++) {
        ELL_spmv_neon_FP32(mtx_esb, vec_b, temp_ell);//只计算到row/VECTOR_LENGTH*VECTOR_LENGTH部分
    }
    gettimeofday(&t2_neon, NULL);
    VALUE_TYPE time_ell = (t2_neon.tv_sec - t1_neon.tv_sec) * 1000.0 + (t2_neon.tv_usec - t1_neon.tv_usec) / 1000.0;
    printf("neon_ell time cost %lf ms\n",time_ell/epoch);
    //mapping
    for(int i=0;i<mtx512_csr->rownum;i++)
    {
        rec_sell_c_a[map_512[i]] = temp_ell[i];
    }
    //compare
    error_num = 0;
    for(int i=0;i<mtx_csr->rownum;i++) {
        if (abs(rec_cpu[i] - rec_sell_c_a[i]) > 1e-5) {
            if (error_num == 0) {
                printf("error! tid=%d %lf,%lf\n", i, rec_cpu[i], rec_sell_c_a[i]);
            }
            error_num++;
        }
    }
    printf("ell error:%d\n",error_num);
    free(map_512);
    free(temp_ell);
///** ACSR && compare **/
    acsr_mtx *A = CSR2ACSR(mtx_csr);
    gettimeofday(&t1_neon, NULL);
    for(int i=0;i<epoch;i++) {
        ACSR_spmv_neon(rec_acsr, A, vec_b);
    }
    gettimeofday(&t2_neon, NULL);
    VALUE_TYPE time_acsr = (t2_neon.tv_sec - t1_neon.tv_sec) * 1000.0 + (t2_neon.tv_usec - t1_neon.tv_usec) / 1000.0;
    printf("neon_ACSR time cost %lf ms\n",time_acsr/epoch);
    error_num = 0;
    for(int i=0;i<mtx_csr->rownum;i++) {
        if (abs(rec_cpu[i] - rec_acsr[i]) > 1e-5) {
            if (error_num == 0) {
                printf("error! tid=%d %lf,%lf\n", i, rec_cpu[i], rec_acsr[i]);
            }
            error_num++;
        }
    }
    printf("ACSR error:%d\n",error_num);
///** symmetry_mtx && compare **/
    symmetry_mtx *sy_mtx = (symmetry_mtx *)malloc(sizeof(symmetry_mtx));
    //trans_symmetry_mtx(mtx_csr,sy_mtx);
    trans_asymmetry_mtx(mtx_csr,sy_mtx);
    gettimeofday(&t1_neon, NULL);
    for(int i=0;i<epoch;i++) {
        //sy_SpMV(rec_sy,sy_mtx,vec_b);
        asy_SpMV(rec_sy, sy_mtx, vec_b);
    }
    gettimeofday(&t2_neon, NULL);
    VALUE_TYPE time_sy = (t2_neon.tv_sec - t1_neon.tv_sec) * 1000.0 + (t2_neon.tv_usec - t1_neon.tv_usec) / 1000.0;
    printf("sy_mtx time cost %lf ms\n",time_sy/epoch);
//    if(sy_mtx->rownum & 1)
//    {
//        rec_sy[sy_mtx->rownum - 1] += mtx_csr->val[mtx_csr->nnznum - 1] * vec_b[sy_mtx->rownum - 1];
//    }
    error_num = 0;
    for(int i=0;i<mtx_csr->rownum;i++)
    {
        if(abs(rec_cpu[i] - rec_sy[i]) > 1e-5)
        {
            if(error_num == 0) {
                printf("error! tid=%d %lf,%lf\n", i, rec_cpu[i], rec_sy[i]);
            }
            error_num++;
        }
    }
    printf("symtx error:%d\n",error_num);
    gettimeofday(&t1_neon, NULL);
    for(int i=0;i<epoch;i++) {
        kml_sparse_scsrgemv(KML_SPARSE_OPERATION_NON_TRANSPOSE, mtx_csr->rownum,
                            mtx_csr->val,
                            mtx_csr->rowptr,
                            mtx_csr->colidx,
                            vec_b,
                            rec_kml);
    }
    gettimeofday(&t2_neon, NULL);
    VALUE_TYPE time_kml = (t2_neon.tv_sec - t1_neon.tv_sec) * 1000.0 + (t2_neon.tv_usec - t1_neon.tv_usec) / 1000.0;
    printf("kml time cost %lf ms\n",time_kml/epoch);

    error_num = 0;
    for(int i=0;i<mtx_csr->rownum;i++)
    {
        if(abs(rec_cpu[i] - rec_kml[i]) > 1e-5)
        {
            if(error_num == 0) {
                printf("error! tid=%d %lf,%lf\n", i, rec_cpu[i], rec_kml[i]);
            }
            error_num++;
        }
    }
    printf("kml error:%d\n",error_num);
    printf("partition:CSR-ELL-ACSR-SY[%d,%d,%d,%d]\n",mtx_csr->nnznum,mtx_esb->nnz,A->vecnum * VECTOR_LENGTH,sy_mtx->vecnum*2);



    /** 写入 **/
    std::string filename = "/home/hw_EMatrix/seery/src/log/result/F32_asy.txt";
    std::string mtx_path(argv[1]);
    std::string prefix = "/home/data/suitesparse_collection/";
    mtx_path.erase(mtx_path.find(prefix), prefix.length());

    std::ofstream outFile(filename, std::ios::app);
    outFile << mtx_path << " " << mtx_csr->rownum << " " << mtx_csr->colnum << " " << mtx_csr->nnznum << " "
    << time_cpu << " " << time_csr << " " << time_ell <<  " " << time_acsr << " " << time_sy << " " << time_kml
    << " " << mtx_csr->nnznum << " " << mtx_esb->nnz << " " << A->vecnum * VECTOR_LENGTH << " " << sy_mtx->vecnum*2 << "\n";
    outFile.close();

    free(mtx_csr);
    free(A);
    free(sy_mtx);
    free(mtx_esb);
    free(vec_b);
    free(rec_cpu);
    free(rec_csr);
    free(rec_sell_c_a);
    free(rec_acsr);
    free(rec_sy);
    free(rec_kml);
}