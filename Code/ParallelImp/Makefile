# Makefile for seam_carving on macOS using Homebrew LLVM and libomp

# Use the Homebrew-installed LLVM clang
CXX = clang++
CXXFLAGS = -std=c++11 -O2 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include `pkg-config --cflags opencv4`
LDFLAGS = -Xpreprocessor -fopenmp -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib -lomp `pkg-config --libs opencv4`

TARGET = seam_carving
SRC = seam_carving.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
