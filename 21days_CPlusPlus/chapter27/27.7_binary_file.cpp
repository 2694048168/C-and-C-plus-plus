#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

struct Human
{
  Human(){};
  Human(const char* inName, int inAge, const char* inDOB) : age(inAge)
  {
    strcpy(name, inName);
    strcpy(DOB, inDOB);
  }

  char name[30];
  int age;
  char DOB[20];
};

int main(int argc, char** argv)
{
  Human input("Siddhartha Rao", 101, "May 1916");

  std::ofstream fsOut ("./MyBinary.bin", std::ios_base::out | std::ios_base::binary);
  if (fsOut.is_open())
  {
    std::cout << "Writing one object of Human to a binary file." << std::endl;
    fsOut.write(reinterpret_cast<const char*>(&input), sizeof(input));
    fsOut.close();
  }

  std::ifstream fsIn ("./MyBinary.bin", std::ios_base::in | std::ios_base::binary);
  if (fsIn.is_open())
  {
    Human somePerson;
    fsIn.read((char*)&somePerson, sizeof(somePerson));
    std::cout << "Reading information from binary file: " << std::endl;
    std::cout << "Name = " << somePerson.name << std::endl;
    std::cout << "Age = " << somePerson.age << std::endl;
    std::cout << "Date of Birth = " << somePerson.DOB << std::endl;
  }

  return 0;
}

// $ g++ -o main 27.7_binary_file.cpp 
// $ ./main.exe 

// Writing one object of Human to a binary file.
// Reading information from binary file:
// Name = Siddhartha Rao
// Age = 101
// Date of Birth = May 1916