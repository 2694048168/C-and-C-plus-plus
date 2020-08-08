/**
* Copyright (c) https://2694048168.github.io
* FileName main.cpp
* Author 黎为
* Description 主函数作为程序的入口
*/

#include "Student.h"
#include  <string.h>

//Student类的构造函数
Student::Student() {
	name[20] = {};
	age = 0;
	sex[10] = {};
}
//Student类的构造函数重载
Student::Student(char name[], int age, char sex[]) {
    //已知strcpy函数的原型是：char *strcpy(char *dst, const char *src);
	strcpy(this->name, name);
	this->age = age;
	strcpy(this->sex, sex);
}
char* Student::getname() {
	return this->name;
}
void Student::putname(char name[]) {
	strcpy(this->name, name);
}
int Student::getage() {
	return age;
}
void Student::putage(int age) {
	this->age = age;
}
char* Student::getsex() {
	return this->sex;
}
void Student::putsex(char sex[]) {
	strcpy(this->sex, sex);
}
//类的析构函数
Student::~Student()
{
}
