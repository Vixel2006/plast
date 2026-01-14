# Compilers
CC   = cc
NVCC = nvcc

# Flags
CFLAGS  = -O3 -Wall -march=native -fopenmp -DCUDA_AVAILABLE
NVFLAGS = -O3 -G -arch=sm_80
DEBUG = -g

# includes
INCLUDES = -I./include -I./include/kernels/cuda -I/usr/local/cuda/include -I./include/optimizers

# Sources
C_SOURCES  = $(wildcard src/*.c) $(wildcard src/kernels/cpu/*.c) $(wildcard src/optimizers/cpu/*.c) $(wildcard *.c)
CU_SOURCES = $(wildcard src/*.cu) $(wildcard src/kernels/cuda/*.cu) $(wildcard src/optimizers/cuda/*.cu)

# Objects
C_OBJS  = $(C_SOURCES:.c=.c.o)
CU_OBJS = $(CU_SOURCES:.cu=.cu.o)

# Target
TARGET = plast

all: $(TARGET)

# Link with nvcc
$(TARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(C_OBJS) $(CU_OBJS) -o $@ -lgomp

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

# Compile CU files
%.o: %.cu
	$(NVCC) $(NVFLAGS) $(DEBUG) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(C_OBJS) $(CU_OBJS)

.PHONY: all clean
