/**
* Copyright (c) 云主宰苍穹
* FileName main.cpp
* Author 黎为
* Description main主函数作为整个程序的入口
*/

#include <iostream.h>
#include <string.h>
#include "stuFile.h"

using namespace std;
//指明函数所在的命名空间namespace
using demo1::cue;
using demo1::addStu;
using demo1::dispFile;

int main(int argc, char *argv[])
{
	int num;
	cue();//格式化输出提示信息
	cin >> num;

	//循环进行学生信息的增删查改操作
	while (num!=0)
	{
		if (num < 0 || num>5) {
			cout << "输出错误，请重新输入！";
			cue();
			cin >> num;
		}
		else {
			char name[20] = {};
			int age=0;
			char sex[10] = {};

			switch (num)
			{
			case 1:
			{
				cout << "请输入学生信息" << endl;
				cout << "学生姓名：";
				cin >> name;
				cout << "学生年龄：";
				cin >> age;
				cout << "学生性别：";
				cin >> sex;
				Student newStu(name,age,sex);
				addStu(newStu);
				cout << "添加成功" << endl;
				system("pause");
				break;
			}
			case 2:
			{
				int num2;
				int n;
				cout << "******************************" << endl;
				cout << "* 1、按姓名删除：            *" << endl;
				cout << "* 2、按年龄删除：            *" << endl;
				cout << "* 3、按性别删除：            *" << endl;
				cout << "* 4、按姓名、年龄、性别删除：*" << endl;
				cout << "******************************" << endl;
				cin >> num2;
				switch (num2)
				{
				case 1:
				{
					cout << "按姓名删除学生：" << endl;
					cin >> name;
					n = 1;
					break;
				}
				case 2:
				{
					cout << "按照年龄删除学生：" << endl;
					cin >> age;
					n = 2;
					break;
				}
				case 3:
				{
					cout << "按性别删除学生：" << endl;
					cin >> sex;
					n = 3;
					break;
				}
				case  4:
				{
					cout << "学生姓名：";
					cin >> name;
					cout << "学生年龄：";
					cin >> age;
					cout << "学生性别：";
					cin >> sex;
					n = 4;
					break;
				}
				default:
					break;
				}
					Student delStu2(name, age, sex);
					demo1::delStu(delStu2,n);
					cout << "成功删除" << endl;
					system("pause");
				break;
			}
			case 3:
			{
				int num3;
				cout << "********************" << endl;
				cout << "* 1、按姓名查找：  *" << endl;
				cout << "* 2、按年龄查找：  *" << endl;
				cout << "* 3、按性别查找：  *" << endl;
				cout << "********************" << endl;
				cin >> num3;
				switch (num3)
				{
				case 1:
				{
					cout << "输入要查找的姓名：" << endl;
					cin >> name;
					Student newStu1;
					newStu1.putname(name);
					demo1::seleStu(newStu1);
					break;
				}
				case 2:
				{
					cout << "输入要查找的年龄：" << endl;
					cin >> age;
					Student newStu2;
					newStu2.putage(age);
					demo1::seleStu(newStu2);
					break;
				}
				case 3:
				{
					cout << "输入要查找的性别：" << endl;
					cin >> sex;
					Student newStu3;
					newStu3.putsex(sex);
					demo1::seleStu(newStu3);
					break;
				}
				default:
					break;
				}
				system("pause");
				break;
			}
			case 4:
			{
				int num4;
				char newName[20] = {};
				int newage = 0;
				char newsex[10] = {};
				cout << "******************************" << endl;
				cout << "*  1、根据学生姓名更改信息： *" << endl;
				cout << "*  2、标准更改学生信息方式： *" << endl;
				cout << "******************************" << endl;
				cin >> num4;
				switch (num4)
				{
				case 1:
				{
					cout << "输入该学生原姓名：" << endl;
					cin >> name;
					cout << "输入学生新姓名：" << endl;
					cin >> newName;
					break;
				}
				case 2:
				{
					cout << "请输入原学生信息" << endl;
					cout << "学生姓名：";
					cin >> name;
					cout << "学生年龄：";
					cin >> age;
					cout << "学生性别：";
					cin >> sex;

					cout << "请输入该学生新信息：" << endl;
					cout << "学生姓名：";
					cin >> newName;
					cout << "学生年龄：";
					cin >> newage;
					cout << "学生性别：";
					cin >> newsex;
					break;
				}
				default:
					break;
				}
				Student stu(name, age, sex);
				Student newStu(newName, newage, newsex);
				demo1::changeStu(stu, newStu, num4);
				cout << "成功修改学生信息：" << endl;
				system("pause");
				break;
			}
			case 5:
				demo1::displayStus();
				system("pause");
			default:
				break;
			}
			cout << endl;
			cue();
			cin >> num;
		}
	}
	return 0;
}
