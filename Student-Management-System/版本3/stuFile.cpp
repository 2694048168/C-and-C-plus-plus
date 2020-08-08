/**
* Copyright (c) https://2694048168.github.io
* FileName main.cpp
* Author 黎为
* Description 主函数作为程序的入口
*/

#include "stuFile.h"
#include <string.h>

void demo1::cue() {
	system("cls");
	printf("***********欢迎使用学生信息管理系统（当前共有%02d名学生）***********\n", dispFile());
	printf("*                                                                *\n");
	printf("*                ---------------------------------               *\n");
	printf("*                | https://2694048168.github.io |                *\n");
	printf("*                ---------------------------------               *\n");
	printf("*                                                                *\n");
	printf("*                1)添加学生信息                                  *\n");
	printf("*                2)删除学生信息                                  *\n");
	printf("*                3)查询学生信息                                  *\n");
	printf("*                4)修改学生信息                                  *\n");
	printf("*                5)显示所有学生信息以及统计信息                  *\n");
	printf("*                0)退出软件                                      *\n");
	printf("*                                                                *\n");
	printf("******************************************************************\n");
}
//添加学生信息
void demo1::addStu(Student stu) {
	//打开文件
	ofstream examplefile("student.txt", ios::app|ios::binary);
	if (examplefile.is_open())
	{
		//向文件中写入stu学生信息
		examplefile.write((char*)&stu,sizeof(stu));

	}
	else {
		cout << "存储失败" << endl;
	}
	//关闭文件
	examplefile.close();
}
//查询指定学生信息
void demo1::seleStu(Student stu) {
	ifstream examplefile("student.txt", ios::in | ios::binary);
	if (examplefile.is_open()) {
		Student evastu;
		int i = 0;
		printf("***************************************\n");
		printf("*----姓名--------年龄--------性别-----*\n");
		while (examplefile.read((char*)&evastu, sizeof(stu))) {
			if (stu.getname() && !strcmp(evastu.getname(), stu.getname())) {
				i++;
				printf("*%8s%12d%12s     *\n", evastu.getname(), evastu.getage(), evastu.getsex());
			}
			else  if (stu.getage() && (stu.getage() == evastu.getage())) {
				i++;
				printf("*%8s%12d%12s     *\n", evastu.getname(), evastu.getage(), evastu.getsex());
			}
			else if (stu.getsex() && !strcmp(evastu.getsex(), stu.getsex())) {
				i++;
				printf("*%8s%12d%12s     *\n", evastu.getname(), evastu.getage(), evastu.getsex());
			}
		}
		examplefile.close();
		printf("***************************************\n");
		cout << "共查找到" << i << "项数据"<< endl;
	}
}
//删除指定学生信息
void demo1::delStu(Student stu,int n) {
	fstream samplefile("student.txt", ios::in | ios::out | ios::binary);
	Student evastu;
	if (samplefile.is_open()) {
		//建立并打开新的空文件
		ofstream examplefile("copyStu.txt", ios::out | ios::binary);
		if (examplefile.is_open()) {
			//将除stu之外的所有学生信息全部读入新文件中
			while (samplefile.read((char*)&evastu, sizeof(evastu)))
			{
				if (n == 1) {
					if (strcmp(evastu.getname(), stu.getname())) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
				}
				else if (n == 2) {
					if (evastu.getage() != stu.getage()) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
				}
				else if (n == 3) {
					if (strcmp(evastu.getsex(), stu.getsex())) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
				}
				else if (n == 4) {
					if ((evastu.getage() != stu.getage()) || strcmp(evastu.getname(), stu.getname()) || strcmp(evastu.getsex(), stu.getsex())) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
				}
			}
			//关闭两个文件
			examplefile.close();
			samplefile.close();
		}
	}
	//将新文件中的数据再拷贝到原文件中
	ofstream oldfile("student.txt", ios::trunc | ios::out | ios::binary);
	ifstream cpfile("copyStu.txt", ios::in | ios::binary);
	if (oldfile.is_open() && cpfile.is_open()) {
		while (cpfile.read((char*)&evastu, sizeof(evastu))) {
			oldfile.write((char*)&evastu, sizeof(evastu));
		}
	}
	cpfile.close();
	oldfile.close();
}
//修改学生信息
void demo1::changeStu(Student stu, Student newStu,int n) {
	fstream samplefile("student.txt", ios::in | ios::out | ios::binary);
	Student evastu;
	if (samplefile.is_open()) {
		ofstream examplefile("copyStu.txt", ios::out | ios::binary);
		if (examplefile.is_open()) {
			while (samplefile.read((char*)&evastu, sizeof(evastu)))
			{
				if (n == 1) {
					if (strcmp(evastu.getname(), stu.getname())) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
					else {
						newStu.putage(evastu.getage());
						newStu.putsex(evastu.getsex());
						examplefile.write((char*)&newStu, sizeof(newStu));
					}
				}
				else if (n == 2) {
					if ((evastu.getage() != stu.getage()) || strcmp(evastu.getname(), stu.getname()) || strcmp(evastu.getsex(), stu.getsex())) {
						examplefile.write((char*)&evastu, sizeof(evastu));
					}
					else {
						examplefile.write((char*)&newStu, sizeof(newStu));
					}
				}
			}
			examplefile.close();
			samplefile.close();
		}
	}

	ofstream oldfile("student.txt", ios::trunc | ios::out | ios::binary);
	ifstream cpfile("copyStu.txt", ios::in | ios::binary);
	if (oldfile.is_open() && cpfile.is_open()) {
		while (cpfile.read((char*)&evastu, sizeof(evastu))) {
			oldfile.write((char*)&evastu, sizeof(evastu));
		}
	}
	cpfile.close();
	oldfile.close();


}
//计算文件中所有学生的数量
int demo1::dispFile() {
	int StuNum = 0;
	ifstream xamplefile("student.txt",ios::in|ios::binary);

	Student stu;
	while (xamplefile.read((char*)&stu, sizeof(stu))) {
		StuNum++;
	}
	xamplefile.close();
	return StuNum;
}
//输出所有学生的信息
void demo1::displayStus(){
	printf("***********当前共有%02d名学生************\n", dispFile());
	printf("*----姓名--------年龄--------性别-----*\n");
	ifstream xamplefile("student.txt", ios::in | ios::binary);

	Student stu;
	while (xamplefile.read((char*)&stu, sizeof(stu))) {
		printf("*%8s%12d%12s     *\n",stu.getname(),stu.getage(),stu.getsex());
	}
	printf("***************************************\n");
	xamplefile.close();
}
