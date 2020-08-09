#include <iostream>
#include <string>

// undefined reference to `WinMain@16'
// error: ld returned 1 exit status
int my_main(int argc, char **argv) 
{
	std::string userName;

	std::cout << "Please enter your first name: ";

	std::cin >> userName;

	std::cout << std::endl 
              << "Hello, " << userName 
              << "\n and goodbye!" << std::endl;

    return 0;
}