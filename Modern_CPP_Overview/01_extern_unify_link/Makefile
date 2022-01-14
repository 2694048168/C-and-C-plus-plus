# C = gcc
# CXX = g++
C = clang
CXX = clang++

SOURCE_C = add.c
OBJECTS_C = add.o

SOURCE_CXX = extern_cpp.cpp

TARGET = extern_cpp
LDFLAGS_COMMON = -std=c++2a

all:
	$(C) -c $(SOURCE_C)
	$(CXX) $(SOURCE_CXX) $(OBJECTS_C) $(LDFLAGS_COMMON) -o $(TARGET)

clean:
	rm -rf *.o $(TARGET)