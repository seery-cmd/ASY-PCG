//
// Created by lenovo on 2024/9/10.
//
#include "readMtx/mmio_highlevel.h"

int main(int argc,char **argv){
    sparse_mtx mtx;
    char *path=argv[1];
    readmtx(path,&mtx);
    printf("rowNum=%d,colNum=%d,nnzNum=%d",mtx.rownum,mtx.colnum,mtx.nnznum);
}

