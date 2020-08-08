/**
* Copyright (c) https://2694048168.github.io
* FileName main.h
* Author 黎为
* Description 作为main函数的头文件
*/

#ifndef STUFILE_H_INCLUDED
#define STUFILE_H_INCLUDED

#include "Student.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace std;
//定义一个demo1命名空间
namespace demo1 {
	//格式化输出提示信息
	void cue();
	//添加学生信息
	void addStu(Student stu);
	//查询学生信息
	void seleStu(Student stu);
	//删除学生信息
	void delStu(Student stu,int n);
	//修改学生信息
	void changeStu(Student stu,Student newStu,int n);
	//输出文件中学生信息的数量
	int dispFile();
	//输出文件中所有学生的详细信息
	void displayStus();
}

#endif // STUFILE_H_INCLUDED
