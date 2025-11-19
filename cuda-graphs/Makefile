# Makefile for CUDA Graph Demonstration
# 
# Usage:
#   make          - Compile the CUDA program
#   make run      - Compile and run the program
#   make clean    - Remove compiled binaries

# Compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_70 -lineinfo
TARGET = cuda_graph_demo
SOURCE = cuda_graph_demo.cu

# Default architecture (adjust based on your GPU)
# Common architectures:
#   sm_70: V100, TITAN V
#   sm_75: RTX 2080, TITAN RTX
#   sm_80: A100, A30
#   sm_86: RTX 3090, A40
#   sm_89: RTX 4090, L40

# Detect architecture automatically (requires nvidia-smi)
ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | sed 's/\.//g' | sed 's/^/sm_/')
ifneq ($(ARCH),sm_)
	CUDA_FLAGS = -O3 -arch=$(ARCH) -lineinfo
endif

# Build target
all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(SOURCE)..."
	@echo "Using CUDA flags: $(CUDA_FLAGS)"
	$(NVCC) $(CUDA_FLAGS) $(SOURCE) -o $(TARGET)
	@echo "Build successful! Executable: $(TARGET)"

# Run the program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@echo ""
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "CUDA Graph Demo Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Compile the CUDA program"
	@echo "  make run      - Compile and run"
	@echo "  make clean    - Remove compiled binaries"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Architectures (modify ARCH in Makefile if needed):"
	@echo "  sm_70: V100, TITAN V"
	@echo "  sm_75: RTX 2080, TITAN RTX"
	@echo "  sm_80: A100, A30"
	@echo "  sm_86: RTX 3090, A40"
	@echo "  sm_89: RTX 4090, L40"

.PHONY: all run clean help
