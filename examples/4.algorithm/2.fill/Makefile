


TARGET=compile_output
SRC=$(shell find . -name "*.cu")
OBJ=$(SRC:.cu=.o)


$(TARGET): $(OBJ)
	nvcc $^ -o $@   --compiler-options -Wall --compiler-options -Wextra --compiler-options -Wconversion --compiler-options -fmax-errors=1 --compiler-options -fno-rtti   -std=c++11 --expt-extended-lambda   -O2

%.o: %.cu
	nvcc -c   --compiler-options -Wall --compiler-options -Wextra --compiler-options -Wconversion --compiler-options -fmax-errors=1 --compiler-options -fno-rtti   -std=c++11 --expt-extended-lambda    -I .  -DXPDL_NUM_OF_GPUS=1 -D_FORCE_INLINES -I../../../ -O2  $^ -o $@


run: 
	./$(TARGET)

allclean:
	-@rm *.o $(TARGET)
