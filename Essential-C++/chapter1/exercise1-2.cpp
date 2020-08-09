#include <iostream>
// #include <string>
// using namespace std;

int main(int argc, char **argv)
{
	string userName;  // error: 'string' was not declared in this scope;

	cout << "Please enter your first name: ";  // error: 'cout' was not declared in this scope;

	cin >> userName;  // error: 'cin' was not declared in this scope;

	cout << endl  // error: 'endl' was not declared in this scope;
         << "Hello, " << userName 
         << "\n and goodbye!" << endl;  // error: 'endl' was not declared in this scope;

    return 0;
}