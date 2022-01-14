# Compile Command for 01_extern_unify_link

## Command line with g++ or clang++

```shell
# step 1. First compile the C code with gcc or clang
gcc -c add.c -o add.o
clang -c add.c -o add.o

# step 2. Second compile the C++ code and link with C compiled code
g++ extern_cpp.cpp add.o -std=c++2a -o extern_cpp
clang++ extern_cpp.cpp add.o -std=c++2a -o extern_cpp
```

## Makefile to compile (make command)

```Makefile
# NOTE: Indentation in Makefile is a tab instead of a space character.
C = gcc
CXX = clang++

SOURCE_C = add.c
OBJECTS_C = add.o

SOURCE_CXX = extern_cpp.cpp

TARGET = extern_cpp
LDFLAGS_COMMON = -std=c++2a

all:
    $(C) -c $(SOURCE_C)
    $(CXX) -c $(SOURCE_CXX) $(OBJECTS_C) $(LDFLAGS_COMMON) -o $(TARGET)

clean:
    rm -rf *.o $(TARGET)
```