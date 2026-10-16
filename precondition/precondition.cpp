//
// Created by lenovo on 2023/9/15.
//
//
#include <stdio.h>
#include "../readMtx/mmio_highlevel.h"
#include "operator_acsr+neon.h"
#include "pcg_acsr+neon.h"
#include <sys/time.h>
#include <fstream>
#include "iostream"

static string matrix_name; //矩阵名
static int mtx_nnz_al; //非零元
static double None_time; //none时间
static double Jacobi_time; //Jacobi时间
static double IC_time; //ICCG时间
static double SSOR_time; //ssor时间
static double matrix_format_trans_time_all = 0.0; //转换时间
static double SSOR_factor; //ssor因子
static int data_type_ = VALUE_TYPE_WIDTH; //数据类型

using namespace std;
void write_in(char *mtx,int nnz,int dimension,double sparsity,double average_weight,int max_degree,int select_mtx);
double PCG_time(char *path,int precon)
{
    double tolerance = 1e-5;
    int max_iter = 10000;
    char* input = path;
    sparse_mtx sparse_A;
    readmtx(input, &sparse_A);
    struct timeval t3, t4;
    gettimeofday(&t3,NULL);
    acsr_mtx* A = CSR2ACSR(&sparse_A);
    gettimeofday(&t4,NULL);
    double Trans_time = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    VALUE_TYPE* u = (VALUE_TYPE*)malloc(A->rownum * sizeof(VALUE_TYPE));
    VALUE_TYPE* X = (VALUE_TYPE*)malloc(A->rownum * sizeof(VALUE_TYPE));
    for (int i = 0; i < A->rownum; i++)
    {
        u[i] = 1.0;
        X[i] = 0.0;
    }
    VALUE_TYPE* B = (VALUE_TYPE*)malloc(A->rownum * sizeof(VALUE_TYPE));
    /**一次SPMV的时间开销**/
    // struct timeval t1, t2;
    // gettimeofday(&t1, NULL);
    ACSR_spmv_neon(B, A, u);
    // gettimeofday(&t2, NULL);
    // double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("SPMV 的时间开销：%lf ms\n", time);
    int mode = precon;
    /**
     * Cholesky要线进行一个对角线元素的处理,乘不同的元素
     * ILU和Block-Jacobi未实现,其中Block-Jacobi为电磁项目上的作业
     */
    if(mode==PCG_IM_CHOLESKY) {
        PCG_Matrix_mulx(sparse_A);//增加正定性能在1.05-1.5之间
    }
    /**
     * 这里得加入一个返回
     */
    Set_sparse_Matrix(sparse_A);

    PCG_setPreconditionerMode_neon(A, (preconditionerID)mode);
    if(get_pds()==1)
    {
        //ICCG分解失败
        return -1.0;
    }//ICCG分解失败
//    cout<<"测试矩阵="<<input<<"\t维数"<<sparse_A.rownum<<"\tVALUE TYPE="<<VALUE_TYPE_WIDTH<<endl;
//    cout<<"转换时间time1="<<Trans_time<<endl;
    matrix_format_trans_time_all += Trans_time;//最后要除以total
    mtx_nnz_al = sparse_A.nnznum;
    matrix_name = input;

//    double Pre_time=Get_Pre_time();//输出预处理时间
//    switch(mode)
//    {
//        case 0:cout<<"None";break;
//        case 1:cout<<"Jacobi";break;
//        case 2:cout<<"In Cholesky";break;
//        case 3:cout<<"SSOR";break;
//    }
//    cout<<"预处理时间time2=:"<<Pre_time<<endl;
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    int iter =PCG_solver_neon(X, B, A->rownum, tolerance, max_iter);
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    if (iter >= max_iter)
    {
        printf("PCG did not converge %d iterations with tolerance %e\n", max_iter, tolerance);
    }
    for (int i = 0; i <10; i++)
    {
        printf("%lf\n", X[i]);
    }
    cout<<"使用了ACSR+全算子NEON+预处理技术"<<endl;
    if(mode==2)
    {
        cout<<"ICCG扩大因子:"<<get_ICCG_add()<<endl;
    }
    printf("iter: %d 次   求解时间time3: %lf ms\n", iter, time);
    cout<<"------------------------------------"<<endl;
    /**
     * 将time写入txt中
     */
    PCG_clean_neon();
    free(X);
    free(B);
    free(u);
    return time;
}
void PCG_run(string path)
{
    int total_num_time=3;//测试10次减少误差
    VALUE_TYPE diag_ic=1.0;//ICCG的对角线扩大
    char *Mtx_path=path.data();
    double time1=0.0;
    double time2=0.0;
    double time3=0.0;
    double time4=0.0;
    int ICCG_time=0;
    for(int see=0;see<total_num_time;see+=1) {
        time1 += PCG_time(Mtx_path, PCG_NONE);
        time2 += PCG_time(Mtx_path, PCG_JACOBI);
        set_pds();
        set_diag(diag_ic);
        double t3=PCG_time(Mtx_path, PCG_IM_CHOLESKY);
        while (t3 < 0) {
            if(ICCG_time>=1000)
            {
                break;
            }//删除大于100次的不能分解的ICCG矩阵
            set_pds();
            diag_ic += 0.1;//扩大increase
            set_diag(diag_ic);
            t3 = PCG_time(Mtx_path, PCG_IM_CHOLESKY);
            ICCG_time+=1;
        }
        if(t3>0) {
            time3 += t3;
        }
        set_pds();
//        diag_ic += 0.1;
//        set_diag(diag_ic);
//        double time3_2 = PCG_time(Mtx_path, PCG_IM_CHOLESKY);
//        if (time3 > time3_2) {
//            time3 = time3_2;
//        }//实验效果来说,在达到可以分解的情况下再扩大一些可以更快收敛,针对suitesparse上的矩阵
        time4 += PCG_time(Mtx_path, PCG_SOR);
    }
    time1/=total_num_time;
    time2/=total_num_time;
    time3/=total_num_time;
    time4/=total_num_time;
//    printf("None:\t%lfms\n",time1);
//    printf("Jacobi:\t%lfms\n",time2);
//    printf("Incomplete_Cholesky:\t%lfms\t\tDiag_increase%lf\n",time3,diag_ic);
//    printf("SSOR:\t%lfms\n",time4);
    None_time = time1;
    Jacobi_time = time2;
    IC_time = time3;
    SSOR_time = time4;
    SSOR_factor = diag_ic;
    matrix_format_trans_time_all /= (total_num_time*4);


//    double min=time1;
//    int mode=1;
//    if(min>time2)
//    {
//        mode=2;
//        min=time2;
//    }
//    if(min>time3&&time3!=0.0)
//    {
//        mode=3;
//        min=time3;
//    }
//    if(min>time4)
//    {
//        mode=4;
//        min=time4;
//    }
//    printf("min_time:\t%lf\t\tselect precon is:\t%d\n",min,mode);


    /**
     * 记录矩阵经典特征
     */
//    sparse_mtx Matrix;
//    readmtx(Mtx_path,&Matrix);
//    double a=Matrix.nnznum;
//    double b=Matrix.rownum;
//    double Sparsity=a/b/b;//矩阵稀疏性
//    int dimension=Matrix.rownum;//维数
//    int nnz=Matrix.nnznum;//非零元个数
//    VALUE_TYPE diag;
//    VALUE_TYPE row_value_all;
//    double average_weight=0.0;//平均对角线占比权重
//    int max_degree=0;//最大出入度
//    for(int i=0;i<Matrix.rownum;i++)
//    {
//        row_value_all=0.0;
//        if(Matrix.rowptr[i+1]-Matrix.rowptr[i]>max_degree)
//        {
//            max_degree=Matrix.rowptr[i+1]-Matrix.rowptr[i];
//        }
//        for(int j=Matrix.rowptr[i];j<Matrix.rowptr[i+1];j++)
//        {
//            row_value_all+=abs(Matrix.val[j]);
//            if(i==Matrix.colidx[j])
//            {
//                diag=abs(Matrix.val[j]);
//            }
//        }
//        average_weight+=diag/row_value_all;
//    }
//    average_weight/=Matrix.rownum;
//    printf("非零元个数:%d\n",nnz);
//    printf("维数:%d\n",dimension);
//    printf("矩阵稀疏性:%lf%%\n",Sparsity*100);
//    printf("平均对角线占比:%lf%%\n",average_weight*100);
//    printf("最大出入度:%d\n",max_degree);
    //write_in(Mtx_path,nnz,dimension,Sparsity,average_weight, max_degree,mode);
    //cout<<"###############################################"<<endl;
}
//void write_in(char *mtx,int nnz,int dimension,double sparsity,double average_weight,int max_degree,int select_mtx)
//{
//    /**
//     * 序号,矩阵名,非零元,维数,稀疏性,对角线平均权重,最大出入度,选择预处理标签
//     */
//    string s;
//    int number=0;//序号
//    ifstream  inf;
//    inf.open("/home/xiaoguoqing/Seery-experiment/src/factor_data/data.txt");
//    while(getline(inf,s))
//    {
//        number++;
//    }
//    inf.close();
//    cout<<"----------------读入完毕-------------------"<<endl;
//    ofstream outf;    //用ofstream类定义输入对象
//    outf.open("/home/xiaoguoqing/Seery-experiment/src/factor_data/data.txt",ios::app);  //文件绝对路径
//    outf<<number<<" "<<mtx<<" "<<nnz<<" "<<dimension<<" "<<sparsity*100<<" "<<average_weight*100<<" "<<max_degree<<" "<<select_mtx-1<<endl;
//    outf.close();
//}
#include <filesystem>
namespace fs = std::filesystem;

bool fileExists(const std::string& path) {
    return fs::exists(path);
}

int main(int argc, char* argv[])
{
//    ifstream file("/home/user/seery/src/data/data-double.txt");
//    if (!file) {
//        cerr << "无法打开文件: " << endl;
//        return 0;
//    }
//    string word;
//    int wordCount = 0;
//    while (file >> word) {
//        if(wordCount % 8 == 1)
//        {
//            PCG_run("/home/user/seery/data" + word);
//            printf("%s %d %lf %lf %lf %lf %lf %lf %d\n",matrix_name.data(),mtx_nnz_al,None_time,Jacobi_time,IC_time,SSOR_time,matrix_format_trans_time_all,SSOR_factor,data_type_);
//        }
//        wordCount++;
//    }
//    file.close();

    PCG_run("/home/user/seery/data/x104/x104.mtx");
    printf("%s %d %lf %lf %lf %lf %lf %lf %d\n",matrix_name.data(),mtx_nnz_al,None_time,Jacobi_time,IC_time,SSOR_time,matrix_format_trans_time_all,SSOR_factor,data_type_);
}