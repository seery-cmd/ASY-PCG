static char help[] = "Test file for the PCG\n";
/*
 * Test file for the PCFactorSetShiftType() routine or -pc_factor_shift_type POSITIVE_DEFINITE option.
 * The test matrix is the example from Kershaw's paper [J.Comp.Phys 1978]
 * of a positive definite matrix for which ILU(0) will give a negative pivot.
 * This means that the CG method will break down; the Manteuffel shift
 * [Math. Comp. 1980] repairs this.
 *
 * Run the executable twice:
 * 1/ without options: the iterative method diverges because of an
 *    indefinite preconditioner
 * 2/ with -pc_factor_shift_positive_definite option (or comment in the PCFactorSetShiftType() line below):
 *    the method will now successfully converge.
 */

#include <petscksp.h>
#include "readMtx/mmio_highlevel.h"
/**
 * 转换精度需要重写编译
 */
int main(int argc, char **argv)
{
    struct timeval t1,t2;
    /** 读取矩阵mtx **/
    sparse_mtx *mtx=(sparse_mtx *)malloc(sizeof(sparse_mtx));
    printf("%s\n",argv[1]);
    readmtx(argv[1],mtx);
    printf("rowNum=%d,colNum=%d,nnzNum=%d\n",mtx->rownum,mtx->colnum,mtx->nnznum);
    VALUE_TYPE *vec_b=(VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx->rownum);
    VALUE_TYPE *rec_b=(VALUE_TYPE *)malloc(sizeof(VALUE_TYPE)*mtx->rownum);
    printf("value数据类型:%ld\n",sizeof(VALUE_TYPE)*8);
    printf("petsc_scala数据类型:%ld\n",sizeof(PetscScalar)*8);
    for(int k=0;k<mtx->rownum;k++)
    {
        vec_b[k]=1.0;
        rec_b[k]=0.0;
    }
    SPMV_csr(mtx,vec_b,rec_b);

    /** 自定义创建 **/
    KSP                ksp;//求解算法
    PC                 pc;
    PetscInt           row,col;
    Vec                my_x,my_b;
    Mat                my_A;
    MPI_Comm           comm;//MPI
    PetscScalar        v;
    KSPConvergedReason reason;//迭代结果
    PetscInt           i, j, its;//int类型
    PetscScalar        *rec_x;
    PetscInt           pet_row_ptr[mtx->rownum];



    PetscFunctionBegin;
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, 0, help));
    comm = PETSC_COMM_SELF;
    row=mtx->rownum,col=mtx->colnum;
    for(i=0;i<mtx->rownum;i++)
    {
        pet_row_ptr[i]=mtx->rowptr[i+1]-mtx->rowptr[i];
    }
    /*
     * Construct the Kershaw matrix
     * and a suitable rhs / initial guess
     */
    /** 自定义的矩阵 **/
    PetscCall(MatCreateSeqAIJ(comm, row, col, 0, pet_row_ptr , &my_A));//创建矩阵,不进行指定,动态的去分配内存,SeqAIJ表示的是CSR存储格式
    PetscCall(VecCreateSeq(comm, col, &my_x));//创建向量
    PetscCall(VecDuplicate(my_x, &my_b));//根据B创新一样的向量X
    v=0.0;
    for(i=0;i<row;i++)
    {
        PetscCall(VecSetValues(my_x, 1, &i, &v, INSERT_VALUES));//1.要添加的向量 2.要添加的元素数目 3.索引下表 4.值 5.覆盖/相加
    }
    for(i=0;i<row;i++)
    {
        v=rec_b[i];
        PetscCall(VecSetValues(my_b, 1, &i, &v, INSERT_VALUES));//1.要添加的向量 2.要添加的元素数目 3.索引下表 4.值 5.覆盖/相加
    }
    for(i=0;i<row;i++)
    {
        for(j=mtx->rowptr[i];j<mtx->rowptr[i+1];j++)
        {
            v=mtx->val[j];
            PetscCall(MatSetValues(my_A, 1, &i, 1, &(mtx->colidx[j]), &v, INSERT_VALUES));//1.添加的矩阵 2.行数 3.行的全局索引 4.列数 5.列的全局索引 6.值
        }
    }
    /** MPI方式组装矩阵 **/
    PetscCall(MatAssemblyBegin(my_A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(my_A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(my_b));
    PetscCall(VecAssemblyEnd(my_b));

    /** 打印矩阵 **/
//    PetscCall(MatView(my_A, PETSC_VIEWER_STDOUT_WORLD));//打印自定义A

    /** 开始求解 **/
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, my_A, my_A));//1.ksp 2.矩阵A 3.要根据矩阵构造预处理矩阵的
    PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, 100000));//设置精度
    /** 求解方法设置 **/
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));//猜测值为1,false则表示从0开始

    /** 预处理设置 **/
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSOR));
    /* PetscCall(PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE)); */
    /** 设置内部数据结构 **/
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSetUp(ksp));

//    /** 查看预处理方法,看预处理矩阵M的一些东西 **/
//    PetscCall(PCFactorGetMatrix(pc, &my_M));//预处理得到的矩阵
//    PetscCall(VecDuplicate(my_b, &my_d));//创建D跟B一样
//    PetscCall(MatGetDiagonal(my_M, my_d));//从M得到对角线元素赋值给D
//    /** 查看x,b **/
//    PetscCall(VecView(my_b,PETSC_VIEWER_STDOUT_WORLD));//全是1
//    PetscCall(PetscPrintf(comm,"--------------------------------\n"));
//    PetscCall(VecView(my_x,PETSC_VIEWER_STDOUT_WORLD));//全是0
    /** 进行求解 **/
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n开始求解\n"));
    gettimeofday(&t1, NULL);
    PetscCall(KSPSolve(ksp, my_b, my_x));
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    PetscCall(KSPGetConvergedReason(ksp, &reason));
    /** 查看x **/
    //PetscCall(VecView(my_x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecGetArray(my_x,&rec_x));
    for(int i=0;i<10;i++)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "number%d=%lf\n",i,rec_x[i]));
    }

    if (reason == KSP_DIVERGED_INDEFINITE_PC) {
        //预处理失败
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nDivergence because of indefinite preconditioner;\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Run the executable again but with -pc_factor_shift_positive_definite option.\n"));
    } else if (reason < 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nOther kind of divergence: this should not happen.\n"));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n没有收敛\n"));
        PetscCall(KSPGetIterationNumber(ksp, &its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nConvergence in %d iterations.\n", (int)its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\ntimecost:%lf ms\n", time));
    } else {
        //求解成功
        PetscCall(KSPGetIterationNumber(ksp, &its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nConvergence in %d iterations.\n", (int)its));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\ntimecost:%lf ms\n", time));
    }

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&my_A));
    PetscCall(VecDestroy(&my_b));
    PetscCall(VecDestroy(&my_x));
    PetscCall(PetscFinalize());
    printf("------------------------------\n");
    return 0;
}
