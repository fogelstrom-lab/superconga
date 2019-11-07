CFLAGS = -I../../BASE/ -O3 -arch=sm_30 --compiler-options="-O3 -fopenmp" -lglut -lGL -lGLU
CFLAGSCPU = -I../../BASE/ -I/usr/local/cuda/include/ -O3 -fopenmp
LFLAGS =-lglut -lGL -lGLU -lboost_system -lboost_filesystem
BOOSTFLAGS =-I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_filesystem-mt

sum_dos:
	nvcc $(CFLAGS) main_sum_dos.cu $(LFLAGS) -o sum_dos

#viewgrid_osx:
#	clang -I../../BASE -lstdc++ -framework OpenGL -framework -GLUT -lm $(BOOSTFLAGS) main_viewgrid.cpp -o viewgrid

