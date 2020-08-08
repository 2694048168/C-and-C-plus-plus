/**
* Copyright (c) https://2694048168.github.io
* FileName main.h
* Author 黎为
* Description 作为main函数的头文件
*/

#ifndef STUDENT_H_INCLUDED
#define STUDENT_H_INCLUDED

#include<string>
#include <iostream>
using namespace std;
//定义demo1命名空间，并将类纳入此空间中
namespace demo1 {
	class Student
	{
	public:
		Student();
		Student(char name[],int age,char sex[]);
		~Student();

	public:
		char* getname();
		void putname(char name[]);
		int getage();
		void putage(int age);
		char* getsex();
		void putsex(char sex[]);
	private:
		char name[20];
		int age;
		char sex[10];
	};
}
using demo1::Student;


#endif // STUDENT_H_INCLUDED
