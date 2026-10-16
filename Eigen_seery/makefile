

PATH_EIGEN=/home/xiaogq/software/Eigen/include
#Eigen头文件路径
MAT_PATH=${test_matrix}
#矩阵路径
log_path=/home/xiaogq/Seery-experiment/Eigen_seery/log/log_data/Eigen_float.log

exe=my_Eigen.cpp
obj=eigen_r

.PHONY : eigen_r e_run


e_run:$(obj)
	./eigen_ ${MAT_PATH} >> $(log_path)
eigen_r:$(exe)
	g++ my_Eigen.cpp -I$(PATH_EIGEN) -o eigen_ -fopenmp
