compiler := nvcc
lib := -lglfw  -lGL -lXrandr -lXi -lXinerama -lX11 -lrt -ldl

all :
	$(compiler) -o main.out glad.c main.cu $(lib)
	./main.out
