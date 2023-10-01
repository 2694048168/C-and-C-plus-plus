CFLAGS := -g -o2 -Wall -Werror -Wno-unused -ldl -std=c++17

target: sub_o add_o main_o main

main:
	g++ $(CFLAGS) sub.o add.o main.o -o main

sub_o:
	g++ $(CFLAGS) -I. -c sub.cpp -o sub.o
	
add_o:
	g++ $(CFLAGS) -I. -c add.cpp -o add.o

main_o:
	g++ $(CFLAGS) -I. -c main.cpp -o main.o

clean:
	rm -rf *.o main
