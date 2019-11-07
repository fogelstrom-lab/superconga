CFLAGS = -I../../BASE/ -O3 -arch=sm_30 --compiler-options="-O3 -fopenmp" -lglut -lGL -lGLU
CFLAGSCPU = -I../../BASE/ -I/usr/local/cuda/include/ -O3 -fopenmp
LFLAGS =-lglut -lGL -lGLU -lboost_system -lboost_filesystem
BOOSTFLAGS =-I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_filesystem-mt

grd2ascii:
	g++ $(CFLAGSCPU) -std=c++0x main_grd2ascii.cpp $(LFLAGS) -o grd2ascii

grd2ascii_osx:
	clang -I../../BASE -lstdc++ -framework OpenGL -framework -GLUT -lm $(BOOSTFLAGS) main_grd2ascii.cpp -o grd2ascii

