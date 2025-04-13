# Makefile for seam_carving.cpp with target "mac"

# C++ compiler
CXX = g++
# Compiler flags: using C++11 standard and optimization, plus OpenCV flags from pkg-config.
CXXFLAGS = -std=c++11 -O2 $(shell pkg-config --cflags opencv4)
# Linker flags: fetched from pkg-config for OpenCV.
LDFLAGS = $(shell pkg-config --libs opencv4)
# Output executable name
TARGET = mac
# Source file(s)
SOURCES = seam_carving.cpp

.PHONY: mac clean

mac: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)