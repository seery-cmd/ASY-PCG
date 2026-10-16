
PATH_MPI=$(MPIPATH)/include
PATH_KML=/usr/local/kml/include
#KML头文件路径
MAT_PATH=/home/xiaogq/Seery-experiment/datasets/Pres_Poisson/Pres_Poisson.mtx
#矩阵路径

ENVIRONMENT=-L/usr/local/kml/lib/kblas/omp -L/usr/local/kml/lib
#使用的是单机版求解器

exe=KML_CG.c
obj=KML_r

.PHONY : KML_r KML_run


KML_run:$(obj)
	./KML_ $(MAT_PATH) >> /home/xiaogq/Seery-experiment/KML/KML_seery/log/KML.log
KML_r:$(exe)
	gcc -I$(PATH_KML) -I$(PATH_MPI) $(ENVIRONMENT) -o KML_ $(exe) -lm -lksolver -lklapack_full -lklapack -lkservice -lkblas -lstdc++ -fopenmp
