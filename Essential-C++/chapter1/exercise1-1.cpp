#include <iostream>
#include <string>

int main(int argc, char **argv)
{
	std::string userName;

	std::cout << "Please enter your first name: ";

	std::cin >> userName;

	std::cout << std::endl 
              << "Hello, " << userName 
              << "\n and goodbye!" << std::endl;

    return 0;
}