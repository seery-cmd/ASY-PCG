//
// Created by lenovo on 2023/9/15.
//

#ifndef PCG_ACSR_NEON_H
#define PCG_ACSR_NEON_H
//
// Created by lenovo on 2023/9/15.
//
#include "../../include/common.h"
#include "operator_acsr+neon.h"
#include "../readMtx/utils.h"
#include "math.h"

typedef enum PREC_ID
{
    PCG_NONE,       // no preconditioner used
    PCG_JACOBI,// Jacobi preconditioner used
    PCG_IM_CHOLESKY, // IC 不完备cholesky分解preconditioner used
    PCG_SOR,   //ILU 不完备LU分解 preconditioner used
    PCG_IM_LU,
    PCG_Block_Jacobi //Block-Jacobi的预处理方式
} preconditionerID;

static preconditionerID precon_mode = PCG_NONE;//预处理方法
static acsr_mtx* default_A = NULL;//ACSR格式矩阵
static sparse_mtx mtx;
/**
 * 静态预存矩阵
 * Incomplete cholesky
 */
static sparse_mtx *Cholesky_L=NULL;//Cholesky分解的L矩阵csr格式
static vector<vector<element_L>> Cholesky_LT;//Cholesky的倒置
static VALUE_TYPE add_diag;//对角线扩大元素
static int ICCG_pds=0;
/**
 * ILU,非对称矩阵所需要的方法，ICCG为ILU的特殊情况
 */
static sparse_mtx *ILU_LU;//LU分解得到的L+U矩阵
/**
 * JACOBI
 */
static VALUE_TYPE* jacobi_array;//jacobi静态数组
/**
 * SOR
 */
static VALUE_TYPE *SOR_array;//sor的对角线元素数组
static VALUE_TYPE *SOR_array_sqrt;//sor的对角线元素数据开根号
static sparse_mtx *SOR_L;
static sparse_mtx *SOR_LT;
static VALUE_TYPE SOR_value=1.0;//松弛因子
//static VALUE_TYPE *tmp;//在SOR的三角求解器中,注释说明，不知道是不是寄存器冲突的问题还是什么，导致结果不对，请注释留意，不过加这个效果不好
/**
 * precon_time预处理时间
 */
double Pre_time_cost=0.0;//预处理时间
double Get_Pre_time()
{
    return Pre_time_cost;
}
void Set_sparse_Matrix(sparse_mtx M)
{
    mtx=M;
}
static void (*precon)(VALUE_TYPE* R, VALUE_TYPE* X, int n) = NULL;//函数指针
void dummy_precon_neon(VALUE_TYPE* R, VALUE_TYPE* X, int n)
{
    memmove((void*)R, (void*)X, n * sizeof(VALUE_TYPE));
}
/**
 * Jacobi预处理
 */
void ACSR_Jacobi_precon_neon(VALUE_TYPE*R, VALUE_TYPE* X, int n)
{
    for (int i = 0; i < n; i++)
    {
        R[i] = X[i] / jacobi_array[i];
    }
}
/**
 * Cholesky的三角求解器
 * 不完备的Cholesky分解预处理
 * 执行Mz=r操作
 */
void ACSR_Imcomplete_Cholesky_precon(VALUE_TYPE *R,VALUE_TYPE *X,int n)
{
    //这里z=R，r=X代替，n表示维数，求解R，X已知
    int start,end;
    VALUE_TYPE temp;
    /**
     * 三角求解下三角
     */
    for(int i=0;i<Cholesky_LT.size();i++)//求第i的值
    {
        temp=0.0;
        for(int i1=0;i1<Cholesky_LT[i].size()-1;i1++)
        {
            temp+=Cholesky_LT[i][i1].data*R[Cholesky_LT[i][i1].row];
        }
        R[i]=(X[i]-temp)/Cholesky_LT[i][Cholesky_LT[i].size()-1].data;
    }
    /**
     * 三角求解上三角
     */
    for(int i=mtx.rownum-1;i>=0;i--)
    {
        temp=0.0;
        start=Cholesky_L->rowptr[i],end=Cholesky_L->rowptr[i+1];
        for(int i1=start+1;i1<end;i1++)
        {
            temp+=Cholesky_L->val[i1]*R[Cholesky_L->colidx[i1]];
        }
        R[i]=(R[i]-temp)/Cholesky_L->val[start];
    }
}
/**
 * SOR的预处理求解器
 */
void ACSR_SOR_precon(VALUE_TYPE *R,VALUE_TYPE *X,int n)
{
    //(*precon)(Z, R, n)，这里z=R，r=X代替，n表示维数，求解R，X已知
    VALUE_TYPE temp;
    for(int i=0;i<mtx.rownum;i++)
    {
        temp=0.0;
        for(int j=SOR_L->rowptr[i];j<SOR_L->rowptr[i+1]-1;j++)
        {
            temp+=SOR_L->val[j]*R[SOR_L->colidx[j]];
        }
        R[i]=(X[i]-temp)/SOR_L->val[SOR_L->rowptr[i+1]-1];
        //tmp[i]=SOR_array[i]*R[i];
        //R[i]*=SOR_array[i];//这里不加tmp数组，输出的结果不一样
    }
    /**
     * 为ARM平台的编译差异，会出现循环合并的情况，属于计算机底层原因
     */
    for(int i=0;i<mtx.rownum;i++)
    {
        R[i]*=SOR_array[i];
    }
    for(int i=mtx.rownum;i>0;i--)
    {
        temp=0.0;
        for(int j=SOR_LT->rowptr[i]-1;j>SOR_LT->rowptr[i-1];j--)
        {
            temp+=SOR_LT->val[j]*R[SOR_LT->colidx[j]];
        }
        R[i-1]=(R[i-1]- temp)/SOR_LT->val[SOR_LT->rowptr[i-1]];
    }
}
//x表示最终值,b表示AX=b,n表示A的维数,tolerance表示残差,max_iter表示最大迭代次数
int PCG_solver_neon(VALUE_TYPE* X, VALUE_TYPE* B, int n, double tolerance, int max_iter)
{
    VALUE_TYPE* P= (VALUE_TYPE*)malloc(n*sizeof(VALUE_TYPE));
    VALUE_TYPE* Q= (VALUE_TYPE*)malloc(n*sizeof(VALUE_TYPE));
    VALUE_TYPE* R = (VALUE_TYPE*)malloc(n*sizeof(VALUE_TYPE));
    VALUE_TYPE* Z = (VALUE_TYPE*)malloc(n*sizeof(VALUE_TYPE));

    VALUE_TYPE r, alpha, beta, s, s_prev;
    VALUE_TYPE t;
    //default_A跟x相乘结果表示为R,X数组初始值都是0
    //执行了r=b-Ax，这里x都为0所以直接赋值
    for(int i=0;i<n;i++)
    {
        R[i]=B[i];
    }
    /** 预处理 **/
    (*precon)(Z, R, n);//选择执行预处理算法,这里就把z求出来了,本来应该是z=M-1*r
    for(int i=0;i<n;i++)
    {
        P[i]=Z[i];
    }
    /** 内积判断 **/
    VALUE_TYPE r0 = vEuklid_neon(R, n);
    if (r0 == 0)
    {
        free(R);
        return 0;
    }
    int iter = 0;
    do
    {
        if (iter > 0)
        {
            s_prev = s;
        }
        else
        {
            vMulv_neon(&s, Z, R, n);//向量乘法
            s_prev = s;
        }
        /** (P,AP)计算 **/
        ACSR_spmv_neon(Q, default_A, P);
        vMulv_neon(&t, P, Q, n);
        /** 标量alpha计算 **/
        alpha = s_prev / t;
        vAddvMuls_neon(X, X, P, alpha, n);//更新x
        vAddvMuls_neon(R, R, Q, -alpha, n);//更新R

        r = vEuklid_neon(R, n);
        iter++;
        if (fabs(r / r0) < tolerance)
        {
            break;
        }
        (*precon)(Z, R, n);
        vMulv_neon(&s, Z, R, n);//向量乘法
        beta = s / s_prev;
        vAddvMuls_neon(P, Z, P, beta, n);//更新R
        /**
         * 残差对比问题
        */
    } while (iter < max_iter);
    free(P);
    free(Q);
    free(R);
    free(Z);
    return iter;
}

void PCG_setPreconditionerMode_neon(void* system_matrix, preconditionerID mode)
{
    struct timeval T3,T4;
    acsr_mtx* A = (acsr_mtx*)system_matrix;
    vector<vector<element_L>> col_stack(mtx.rownum);//存储列的栈,初始都是0
    element_L col_data[mtx.rownum];//收集数组
    int nan_num=0;
    int inf_num=0;
    precon_mode = mode;
    if (A != NULL)
    {
        default_A = A;
    }
    /**
     * 开始选择哪种预处理
     */
    switch (precon_mode)
    {
        case PCG_NONE:
            precon = dummy_precon_neon;
            break;
        case PCG_JACOBI:
            precon = ACSR_Jacobi_precon_neon;
            gettimeofday(&T3, NULL);
            jacobi_array = (VALUE_TYPE*)malloc(mtx.rownum*sizeof(VALUE_TYPE));
            for(int i=0;i<mtx.rownum;i++)
            {
                for(int j=mtx.rowptr[i];j<mtx.rowptr[i+1];j++)
                {
                    if(mtx.colidx[j]==i)
                    {
                        jacobi_array[i]=mtx.val[j];
                    }
                }
            }
            gettimeofday(&T4, NULL);
            Pre_time_cost=(T4.tv_sec - T3.tv_sec) * 1000.0 + (T4.tv_usec - T3.tv_usec) / 1000.0;
            break;
        case PCG_IM_CHOLESKY:
            precon=ACSR_Imcomplete_Cholesky_precon;
            /**
             * step1初始化
             */
            sparse_mtx *new_L;
            int num_of_L;//用以记录L矩阵元素
            num_of_L=0;
            new_L=(sparse_mtx *) malloc(sizeof (sparse_mtx));
            new_L->rownum=mtx.rownum;
            new_L->colnum=mtx.colnum;
            new_L->val=(VALUE_TYPE *)malloc(sizeof (VALUE_TYPE)*mtx.nnznum);
            new_L->colidx=(int *) malloc(sizeof(int)*mtx.nnznum);
            new_L->rowptr=(int *) malloc(sizeof (int)*(mtx.rownum+1));
            new_L->rowptr[0]=0;//row偏移量一开始是0
            /**
             * step1 done
             */
            gettimeofday(&T3, NULL);
            /**
             * step2分解
             */
            int start,end;
            for(int i=0;i<mtx.rownum;i++)
            {
                //第i行分解
                start=mtx.rowptr[i],end=mtx.rowptr[i+1];
                for(int j=start;j<end;j++)
                {
                    if(mtx.colidx[j]<i)
                    {
                        continue;
                    }
                    else if(mtx.colidx[j]==i)//矩阵的行和列相等
                    {
                        /**
                         * 开始求解L的第i列的对角线元素
                         */
                         //cout<<"分解的对角线元素="<<mtx.val[j]<<endl;
                         VALUE_TYPE temp=0.0;
                            /**
                             * 在列中寻找是否存在乘元素
                             */
                         for(int k1=0;k1<col_stack[i].size();k1++)
                         {
                             temp+=col_stack[i][k1].data*col_stack[i][k1].data;
                         }
                         if(mtx.val[j]-temp<=0)
                         {
                             //cout<<"ICCG分解失败\t在第i行:"<<i<<"\ttemp:"<<temp<<"\tval:"<<mtx.val[j]<<"\t对角线扩大值为:"<<add_diag<<endl;//出现M矩阵分解失败情况
                             ICCG_pds=1;
                             break;
                             //
                         }
                         new_L->val[num_of_L]=sqrt(mtx.val[j] - temp);
                         new_L->colidx[num_of_L]=i;
                         element_L new_data;
                         new_data.data=new_L->val[num_of_L];
                         new_data.row=i;
                         col_stack[i].push_back(new_data);
                         num_of_L++;
                    }
                    else//mtx.colidx[j]>i
                    {
                        VALUE_TYPE temp=0.0;
                        //i==first,colidx[j]==second;
                        //cout<<"分解的非对角线元素是,按照行分解"<<mtx.val[j]<<endl;
                        int first_col=0,second_col=0;
                        //需要去两个vector里面找是否有行一样的元素
                        while(first_col<col_stack[i].size()&&second_col<col_stack[mtx.colidx[j]].size()) {
                            if (col_stack[i][first_col].row == col_stack[mtx.colidx[j]][second_col].row) {
                                temp+=col_stack[i][first_col].data*col_stack[mtx.colidx[j]][second_col].data;
                                first_col++;
                                second_col++;
                            }
                            else if(col_stack[i][first_col].row > col_stack[mtx.colidx[j]][second_col].row)
                            {
                                second_col++;
                            }
                            else
                            {
                                first_col++;
                            }
                        }
                        new_L->val[num_of_L]=1.0/col_stack[i][col_stack[i].size()-1].data*(mtx.val[j]-temp);
//这里会出现nan的情况,对应temp里面出现的inf导致
                        new_L->colidx[num_of_L]=mtx.colidx[j];
                        element_L new_data;
                        new_data.data=new_L->val[num_of_L];
                        new_data.row=i;
                        col_stack[mtx.colidx[j]].push_back(new_data);
                        num_of_L++;
                    }
                }
                if(ICCG_pds==1)
                {
                    break;
                }
                new_L->rowptr[i+1]=num_of_L;
            }
            if(ICCG_pds==1)
            {
                break;
            }
            /**
             * step2 done
             */
            gettimeofday(&T4, NULL);
            Pre_time_cost=(T4.tv_sec - T3.tv_sec) * 1000.0 + (T4.tv_usec - T3.tv_usec) / 1000.0;
            new_L->nnznum=num_of_L;//赋值分解的L矩阵非零元
            Cholesky_L=new_L;//赋值静态数组
            Cholesky_LT=col_stack;
            /**
             * nan和inf的error个数输出
             */
            int start_6311,end_6311;
            for(int se=0;se<new_L->rownum;se++)
            {
                start_6311=new_L->rowptr[se],end_6311=new_L->rowptr[se+1];
                for(int se1=start_6311;se1<end_6311;se1++) {
                    if (isnan(new_L->val[se1])) {
                        //cout << new_L->colidx[se1] << "\t" << "row="<<se<<endl;
                        nan_num++;
                    }
                    if (isinf(new_L->val[se1])) {
                        //cout << new_L->colidx[se1] << "\t" << "row="<<se<<endl;
                        inf_num++;
                    }
                }
            }
            //cout<<"nan num:"<<nan_num<<"\t"<<"inf num:"<<inf_num<<endl;
            break;
        case PCG_IM_LU:
            break;
        case PCG_SOR:
            precon = ACSR_SOR_precon;
            gettimeofday(&T3, NULL);
            /**
             * 初始化
             */
            SOR_L=(sparse_mtx *) malloc(sizeof (sparse_mtx));
            SOR_L->rownum=mtx.rownum;
            SOR_L->colnum=mtx.colnum;
            SOR_L->val=(VALUE_TYPE *)malloc(sizeof (VALUE_TYPE)*mtx.nnznum);
            SOR_L->colidx=(int *) malloc(sizeof(int)*mtx.nnznum);
            SOR_L->rowptr=(int *) malloc(sizeof (int)*(mtx.rownum+1));
            SOR_L->rowptr[0]=0;//row偏移量一开始是0
            SOR_LT=(sparse_mtx *) malloc(sizeof (sparse_mtx));
            SOR_LT->rownum=mtx.rownum;
            SOR_LT->colnum=mtx.colnum;
            SOR_LT->val=(VALUE_TYPE *)malloc(sizeof (VALUE_TYPE)*mtx.nnznum);
            SOR_LT->colidx=(int *) malloc(sizeof(int)*mtx.nnznum);
            SOR_LT->rowptr=(int *) malloc(sizeof (int)*(mtx.rownum+1));
            SOR_LT->rowptr[0]=0;//row偏移量一开始是0
            SOR_array=(VALUE_TYPE *) malloc(sizeof(VALUE_TYPE)*mtx.rownum);
            SOR_array_sqrt=(VALUE_TYPE *) malloc(sizeof(VALUE_TYPE)*mtx.rownum);
            int SOR_L_num, SOR_LT_num;
            SOR_L_num=0;//记录元素
            SOR_LT_num = 0;//记录元素
            for(int i=0;i<mtx.rownum;i++)
            {
                for(int j=mtx.rowptr[i];j<mtx.rowptr[i+1];j++)
                {
                    if(mtx.colidx[j]<i)
                    {
                        SOR_L->val[SOR_L_num]=SOR_value*(mtx.val[j]);
                        SOR_L->colidx[SOR_L_num]=mtx.colidx[j];
                        SOR_L_num++;
                    }
                    else if(mtx.colidx[j]==i)
                    {
                        SOR_L->val[SOR_L_num]=mtx.val[j];
                        SOR_L->colidx[SOR_L_num]=mtx.colidx[j];
                        SOR_L_num++;

                        SOR_array[i]=mtx.val[j];

                        SOR_LT->val[SOR_LT_num]=mtx.val[j];
                        SOR_LT->colidx[SOR_LT_num]=mtx.colidx[j];
                        SOR_LT_num++;
                    }
                    else
                    {
                        SOR_LT->val[SOR_LT_num]=SOR_value*(mtx.val[j]);
                        SOR_LT->colidx[SOR_LT_num]=mtx.colidx[j];
                        SOR_LT_num++;
                    }
                }
                SOR_L->rowptr[i+1]=SOR_L_num;
                SOR_LT->rowptr[i+1]=SOR_LT_num;
            }
            gettimeofday(&T4, NULL);
            Pre_time_cost=(T4.tv_sec - T3.tv_sec) * 1000.0 + (T4.tv_usec - T3.tv_usec) / 1000.0;
            break;
        default:
            precon_mode = PCG_NONE;
            precon = dummy_precon_neon;
            printf("warning: no valid preconditioner ID given and none preconditioner set!\n");
    }
}
void PCG_clean_neon()
{
    return;
}
/**
 * 矩阵值对角线元素乘标量
*/
void PCG_Matrix_mulx(sparse_mtx &mtx)
{
    int start,end;
    for(int i=0;i<mtx.rownum;i++) {
        start=mtx.rowptr[i],end=mtx.rowptr[i+1];
        for (int j = start; j < end; j++) {
            if(i==mtx.colidx[j]) {
                mtx.val[j] *= add_diag;
            }
        }
    }
}
void set_diag(VALUE_TYPE diag)
{
    add_diag=diag;
}
int get_pds()
{
    return ICCG_pds;
}
VALUE_TYPE get_ICCG_add()
{
    return add_diag;
}
void set_pds()
{
    ICCG_pds=0;
}
#endif //PCG_PCG_ACSR_NEON_H
