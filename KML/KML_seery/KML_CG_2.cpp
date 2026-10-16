//
// Created by lenovo on 2024/9/12.
//
#include "readMtx/mmio_highlevel.h"
#include <stdio.h>
#include <kml_iss.h>
#include <math.h>
#include <omp.h>
/**
 * 转换精度修改float和double就可以了
 * 后面SII,DII就是区别64精度和32精度的,S--32float，D---64double
 */
int main(int argc,char **argv)
{
    int core_num=omp_get_max_threads();
    printf("鲲鹏920核心数core_num=%d\n",core_num);
    sparse_mtx *mtx=(sparse_mtx *)malloc(sizeof (sparse_mtx));
    readmtx(argv[1],mtx);
    printf("row:%d\tcol:%d\tnnz:%d\n",mtx->rownum,mtx->colnum,mtx->nnznum);
    /** KML求解 **/
    VALUE_TYPE vec_b[mtx->rownum];
    VALUE_TYPE rec_x[mtx->rownum];
    for(int i=0;i<mtx->rownum;i++)
    {
        vec_b[i]=1.0;
        rec_x[i]=0.0;
    }
    SPMV_csr(mtx,vec_b,rec_x);//计算其中b全为1

    KmlSolverTask *handle;
    omp_set_num_threads(1);
    int ierr;//报错信息
    int n = mtx->rownum;//矩阵维数


    ierr = KmlIssCgInitDI(&handle, n, mtx->val, mtx->colidx, mtx->rowptr);
    ierr = KmlIssCgAnalyzeDI(&handle);
    ierr = KmlIssCgFactorizeDI(&handle);

    int nd = 1;
    int idata[] = {0};
    double ddata[] = {0};
    double eps = 1e-5;
    ierr = KmlIssCgSetDID(&handle, KMLSS_THRESHOLD, &eps, nd);
    int max_iters = 100000;
    ierr = KmlIssCgSetDII(&handle, KMLSS_MAX_ITERATION_COUNT, &max_iters, nd);

    int tolerance_compute = KMLSS_L2;
    ierr = KmlIssCgSetDII(&handle, KMLSS_VECTOR_NORM_TYPE, &tolerance_compute, nd);//残差计算范数
    //L1和L2

    int mtx_mode = KMLSS_CSR;
    ierr = KmlIssCgSetDII(&handle, KMLSS_MATRIX_FORMAT, &mtx_mode, nd);//矩阵存储格式

//    int per_condition_mode = KMLSS_ICC;
//    ierr = KmlIssCgSetDII(&handle, KMLSS_PRECONDITIONER_TYPE, &per_condition_mode, nd);//预处理模式

/** 求解 **/
    int ldx = n;
    int ldb = n;
    int nb = 1;

    double x[mtx->rownum];
    for(int i=0;i<mtx->rownum;i++)
    {
        x[i]=0.0;
    }
    struct timeval t1,t2;
    gettimeofday(&t1,NULL);
    ierr = KmlIssCgSolveDI(&handle, nb, x, ldx, rec_x, ldb);
    gettimeofday(&t2,NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
/** 得到结果 **/
    ierr = KmlIssCgGetDII(&handle, KMLSS_ITERATION_COUNT, idata, nd);
    ierr = KmlIssCgGetDID(&handle, KMLSS_TOLERANCE, ddata, nd);

    printf("[Solve info] Res= \t%.12g\t, iteration = \t%d\t\n", sqrt(ddata[0]), idata[0]);
    printf("时间开销timecost:%lf\n",time);

    for(int i=0;i<10;i++)
    {
        printf("%lf\n",x[i]);
    }

    /** 释放内存 **/
    //ierr = KmlIssCgCleanDI(&handle);
    return 0;
}
