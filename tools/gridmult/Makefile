CFLAGS = -I../../BASE/ -O3 -arch=sm_30 --compiler-options="-O3 -fopenmp" -lglut -lGL -lGLU
CFLAGSCPU = -I../../BASE/ -I/usr/local/cuda/include/ -O3 -fopenmp
LFLAGS =-lglut -lGL -lGLU -lboost_system -lboost_filesystem
BOOSTFLAGS =-I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_filesystem-mt

main:
	nvcc $(CFLAGS) main_gridmult.cu $(LFLAGS) -o gridmult
