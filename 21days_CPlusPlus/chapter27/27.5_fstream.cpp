#include <iostream>
#include <fstream>

// 使用 ofstream 新建一个文本文件并向其中写入文本
int main(int argc, char** argv)
{
  std::ofstream myFile;
  myFile.open("./helloFile.txt", std::ios_base::out);
  if (myFile.is_open())
  {
    std::cout << "File open successfully." << std::endl;
    myFile << "My first text file." << std::endl;
    myFile << "Hello File." << std::endl;
    std::cout << "Finished writing to file, will close now." << std::endl;
    myFile.close();
  }

  return 0;
}

// $ g++ -o main 27.5_fstream.cpp 
// $ ./main.exe 

// File open successfully.
// Finished writing to file, will close no
