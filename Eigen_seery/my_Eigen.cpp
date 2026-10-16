//
// Created by lenovo on 2024/9/10.
//

#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/unsupported/Eigen/SparseExtra>
#include "readMtx/mmio_highlevel.h"

#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;
using namespace Eigen;
/**
 * 转换精度修改float和double就可以了
 */
void eigen_run(char *mtx_path,int mode)
{
    /** 读取矩阵mtx **/
    sparse_mtx *mtx=(sparse_mtx *)malloc(sizeof(sparse_mtx));
    char *path=mtx_path;
    readmtx(path,mtx);
    VALUE_TYPE vec_b[mtx->rownum];
    VALUE_TYPE rec_x[mtx->rownum];
    for(int i=0;i<mtx->rownum;i++)
    {
        vec_b[i]=1.0;
        rec_x[i]=0.0;
    }
    SPMV_csr(mtx,vec_b,rec_x);//计算其中b全为1
//    VectorXd b(mtx->rownum);//定义b向量--double类型
//    VectorXd x(mtx->rownum);//定义x向量--double类型
    VectorXf b(mtx->rownum);//定义b向量--float类型
    VectorXf x(mtx->rownum);//定义x向量--float类型
    for(int i=0;i<mtx->rownum;i++)
    {
        b(i)=rec_x[i];
    }
    omp_set_num_threads(1);//设置一个线程跑任务
    //cout<<b<<endl;//打印b向量
    SparseMatrix<VALUE_TYPE,RowMajorBit> Eigen_mat(mtx->rownum,mtx->colnum);//定义稀疏矩阵
    Eigen_mat.reserve(mtx->nnznum);
    vector<Triplet<VALUE_TYPE>> tripletList;//三元组
    tripletList.reserve(mtx->nnznum);
    for(int i=0;i<mtx->rownum;i++)
    {
        for(int j=mtx->rowptr[i];j<mtx->rowptr[i+1];j++)
        {
            tripletList.emplace_back(i, mtx->colidx[j], mtx->val[j]);
        }
    }
    Eigen_mat.setFromTriplets(tripletList.begin(), tripletList.end());//矩阵mtx构造完成

    if(mode==1) {
        ConjugateGradient<SparseMatrix<VALUE_TYPE>, Lower | Upper, IdentityPreconditioner> cg;
        //DiagonalPrecondiiton-----Jacobi预处理,默认
        //IncompleteCholesky------ICCG预处理
        //IdentityPreconditioner----用户自定义预处理
        cg.setMaxIterations(100000);//设置最大迭代次数
        cg.setTolerance(1e-5);//设置误差
        struct timeval t1,t2;
        cg.compute(Eigen_mat);
        gettimeofday(&t1,NULL);
        x = cg.solve(b);
        gettimeofday(&t2,NULL);
        double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        cout << "迭代次数iterations:" << cg.iterations() << endl;
        cout << "误差估计estimated error: " << cg.error()      << endl;
        cout << "solver Info：" <<cg.info() <<endl;
        cout << "数据类型"<< VALUE_TYPE_WIDTH<<endl;

        cout << "预处理方法" <<endl;
        printf("时间开销timecost:%lf ms\n",time);
        cout<<"result:"<<endl;
        for(int i=0;i<10;i++)
        {
            cout<<x(i)<<endl;
        }
        printf("-------------------------------------\n");
    }
    else if(mode==2)
    {
        ConjugateGradient<SparseMatrix<VALUE_TYPE>, Lower | Upper, DiagonalPreconditioner<VALUE_TYPE>> cg;
        //DiagonalPrecondiiton-----Jacobi预处理,默认
        //IncompleteCholesky------ICCG预处理
        //IdentityPreconditioner----用户自定义预处理
        cg.setMaxIterations(100000);//设置最大迭代次数
        cg.setTolerance(1e-5);//设置误差
        struct timeval t1,t2;
        cg.compute(Eigen_mat);
        gettimeofday(&t1,NULL);
        x = cg.solve(b);
        gettimeofday(&t2,NULL);
        double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        cout << "迭代次数iterations:" << cg.iterations() << endl;
        cout << "误差估计estimated error: " << cg.error()      << endl;
        cout << "solver Info：" <<cg.info() <<endl;
        cout << "数据类型"<< VALUE_TYPE_WIDTH<<endl;

        cout << "预处理方法" <<endl;
        printf("时间开销timecost:%lf ms\n",time);
        cout<<"result:"<<endl;
        for(int i=0;i<10;i++)
        {
            cout<<x(i)<<endl;
        }
        printf("-------------------------------------\n");
    }
    else
    {
        ConjugateGradient<SparseMatrix<VALUE_TYPE>, Lower | Upper, IncompleteCholesky<VALUE_TYPE>> cg;
        //DiagonalPrecondiiton-----Jacobi预处理,默认
        //IncompleteCholesky------ICCG预处理
        //IdentityPreconditioner----用户自定义预处理
        cg.setMaxIterations(100000);//设置最大迭代次数
        cg.setTolerance(1e-5);//设置误差
        struct timeval t1,t2;
        cg.compute(Eigen_mat);
        gettimeofday(&t1,NULL);
        x = cg.solve(b);
        gettimeofday(&t2,NULL);
        double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        cout << "迭代次数iterations:" << cg.iterations() << endl;
        cout << "误差估计estimated error: " << cg.error()      << endl;
        cout << "solver Info：" <<cg.info() <<endl;
        cout << "数据类型"<< VALUE_TYPE_WIDTH<<endl;
        printf("时间开销timecost:%lf ms\n",time);
        cout<<"result:"<<endl;
        for(int i=0;i<10;i++)
        {
            cout<<x(i)<<endl;
        }
        printf("-------------------------------------\n");
    }
}


int main(int argc, char **argv)
{
    char *mtx_path=argv[1];
    sparse_mtx *mtx=(sparse_mtx *)malloc(sizeof(sparse_mtx));
    printf("%s\n",mtx_path);
    readmtx(mtx_path,mtx);
    printf("rowNum=%d,colNum=%d,nnzNum=%d\n",mtx->rownum,mtx->colnum,mtx->nnznum);

    cout<<"none预处理:"<<endl;
    eigen_run(mtx_path,1);
    cout<<"Jacobi预处理:"<<endl;
    eigen_run(mtx_path,2);
    cout<<"ICCG预处理:"<<endl;
    eigen_run(mtx_path,3);
}