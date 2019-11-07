CFLAGS_OZ =	-I include/ \
			-O3 -arch=sm_30 --compiler-options="-O3" \
			-lglut -lGL -lGLU \
			-lboost_system -lboost_filesystem \
			-lgfortran -llapack -lblas \
			-std=c++11

demo_hexagon_swave_VL:
	nvcc $(CFLAGS_OZ) demo_hexagon_swave_VL.cu -o demo_hexagon_swave_VL
