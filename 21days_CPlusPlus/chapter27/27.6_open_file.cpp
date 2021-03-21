#include <iostream>
#include <fstream>
#include <string>

// 使用 open() 和运算符读取文本文件
int main(int argc, char** argv)
{
  std::ifstream myFile;
  myFile.open("./helloFile.txt", std::ios_base::in);
  if (myFile.is_open())
  {
    std::cout << "File open successfully. It contains: " << std::endl;
    std::string fileContains;
    while (myFile.good())
    {
      getline(myFile, fileContains);
      std::cout << fileContains << std::endl;
    }

    std::cout << "Finished reading file, will close now." << std::endl;
    myFile.close();
  }
  else
  {
    std::cout << "open() failed: check if file is in right folder." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 27.6_open_file.cpp 
// $ ./main.exe

// File open successfully. It contains:  
// My first text file.
// Hello File.

// Finished reading file, will close now.
