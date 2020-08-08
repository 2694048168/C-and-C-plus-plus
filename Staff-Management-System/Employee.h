/* file_name: Employee.h
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

#ifndef EMPLOYEE_H
#define EMPLOYEE_H


#include <iostream>
#include "Worker.h"
using namespace std;

// 普通员工
class Employee :public Worker
{
public:

	// 显示所有信息
	void showInfo();

	// 获取岗位名称
	string getDeptName();

	// 构造函数
	Employee(int id, string name, int dId);

	// 析构函数
	~Employee();
	
};

#endif // EMPLOYEE_H