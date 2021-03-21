## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 27.9.1 测验
1. 在只需要写入文件的情况下，应该使用那种流？
- ofstream

2. 如何使用 cin 从输入流中获取一整行？
- std::cin.getline(std::cin, inputLine)

3. 在需要将 std::string 对象写入文件时，应使用 ios_bse::binary 模式吗？
- 不应该
- std::string 包含文本信息，应该使用默认的文本文件，文本模式 

4. 在使用 open() 打开流后，为何还要使用 is_open() 进行检查？
- 进行文件操作的前提是文件流正确的被打开了，否则引发错误


### 27.9.2 练习
1. 查错：指出下述代码中的错误：

```C++
fstream myFile;
myFile.open("HelloFile.txt", ios_base::out);
myFile << "Hello file!";
myFile.close();
```

- check file stream
- myFile.is_open()

2. 查错：指出下述代码中的错误：

```C++
ifstream myFile("SomeFile.txt");
if(myFile.is_open())
{
  myFile << "This is some text" << endl;
  myFile.close();
}
```

- ifstream 是用于输入，而不是输出，不支持流插入运算符 <<