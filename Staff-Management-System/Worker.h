/* file_name: Worker.h
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

#ifndef WORKER_H
#define WORKER_H

#include <iostream>
#include <string>
using namespace std;

/* 职工类抽象，利用多态管理不同类型职工（普通职工、经理、老板）
** 职工属性：编号、姓名、部门
** 职工行为：岗位职责信息描述，获取岗位名称
*/

// 职工抽象基类
class Worker
{
public:

	// 显示所有信息
	virtual void showInfo() = 0;

	// 获取岗位名称
	virtual string getDeptName() = 0;

	// 职工编号
	int m_Id;

	// 职工姓名
	string m_Name;

	// 部门编号
	int m_DeptId;

	// 构造函数
	Worker();

	// 析构函数
	virtual ~Worker();
	
};

#endif // WORKER_H
