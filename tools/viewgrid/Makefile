CFLAGS = -I../../BASE/ -O3 -arch=sm_30 --compiler-options="-O3 -fopenmp" -lglut -lGL -lGLU
CFLAGSCPU = -I../../BASE/ -I/usr/local/cuda/include/ -O3 -fopenmp
LFLAGS =-lglut -lGL -lGLU -lboost_system -lboost_filesystem
BOOSTFLAGS =-I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_filesystem-mt

viewgrid:
	nvcc $(CFLAGS) main_viewgrid.cu $(LFLAGS) -o viewgrid

viewgrid_float:
	nvcc $(CFLAGS) main_viewgrid.cu $(LFLAGS) -o viewgrid_float

viewgrid_cpu:
	g++ $(CFLAGSCPU) main_viewgrid.cpp $(LFLAGS) -o viewgrid_cpu

viewgrid_osx:
	clang -I../../BASE -lstdc++ -framework OpenGL -framework -GLUT -lm $(BOOSTFLAGS) main_viewgrid.cpp -o viewgrid

