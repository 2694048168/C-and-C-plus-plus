#include <iostream>

extern bool fibon_elem(int, int&);

int main(int argc, char*argv[])
{
    int pos, elem;
    char ch;
    bool more = true;

    while (more)
    {
        std::cout << "Please enter a position: ";
        std::cin >> pos;

        if (fibon_elem(pos, elem))
        {
            std::cout << "element # " << pos << " is " << elem << std::endl;
        }
        else
        {
            std::cout << "Sorry. Could not calculate element # " << pos << std::endl;
        }
        
        std::cout << "Would you like to try again? (y/n)";
        std::cin >> ch;
        if (ch != 'y' && ch != 'Y')
        {
            more = false;
        }
    }

    return 0;
}