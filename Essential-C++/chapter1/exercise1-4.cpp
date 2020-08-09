#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
	std::vector<std::string> vecName(2);

	std::cout << "Please enter your first name: ";
    std::cin >> vecName[0];
	
    std::cout << "Please enter your last name: ";
	std::cin >> vecName[1];
	
    for (std::vector<std::string>::iterator itr = vecName.begin(), vecEnd = vecName.end(); itr != vecEnd; ++itr)
		std::cout << *itr << " ";
	
    return 0;
}