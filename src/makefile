EXE=main
OBJ=../main.cpp

mtx=nlpkkt120
MTX=/home/data/suitesparse_collection/${mtx}/${mtx}.mtx
LOG_PATH=/home/hw_EMatrix/seery/src/log/test.log

#MTX=${MTX}
#LOG_PATH=${LOG}

.PHONY: run obj clean

run:obj
	./$(EXE) $(MTX)
obj:
	g++ ${OBJ} -I /home/hw_EMatrix/seery/src/software/kml/include -L /home/hw_EMatrix/seery/src/software/kml/lib/neon/kspblas/single -lkspblas -o ${EXE} -O2 -lstdc++ -lm

clean:
	rm $(EXE)