
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

MPI_ROOT=/home/xiaogq/software/petsc-3.21.5/test/bin
path=${test_matrix}
log=/home/xiaogq/Seery-experiment/PETSc/log/log_data/petsc_float_SSOR.log

obj=petsc_r
exe=petsc_test.c

.PHONY : petsc_r p_run

p_run:$(obj)
	${MPI_ROOT}/mpirun -n 1 ./pet $(path) >> $(log)
petsc_r:$(exe)
	${MPI_ROOT}/mpicc -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -Wno-stringop-overflow -fstack-protector -fvisibility=hidden -g3 -O0  -I/home/xiaogq/software/petsc-3.21.5/include -I/home/xiaogq/software/petsc-3.21.5/test/include     -Wl,-export-dynamic petsc_test.c  -Wl,-rpath,/home/xiaogq/software/petsc-3.21.5/test/lib -L/home/xiaogq/software/petsc-3.21.5/test/lib -lpetsc -lmpi -lm  -o pet


