
CXX = g++

CXXFLAGS = -std=c++11 -O2 $(shell pkg-config --cflags opencv4)

LDFLAGS = $(shell pkg-config --libs opencv4)

TARGET = seam_carving

SOURCES = seam_carving.cpp

.PHONY: seam_carving clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)