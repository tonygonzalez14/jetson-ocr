# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -shared -Xcompiler -fPIC

# Source files
SRC = cuda_inference.cu lodepng.cpp

# Output library
LIB = libcuda_inference.so

# Default target
all: $(LIB)

# Build shared library
$(LIB): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Clean up
clean:
	rm -f $(LIB)

# Phony targets
.PHONY: all clean

