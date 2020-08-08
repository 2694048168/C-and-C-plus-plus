/* file_name: Manager.cpp
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

#include <iostream>
#include "Manager.h"
#include <string>
using namespace std;

// 显示所有信息
void Manager::showInfo()
{
    cout << "职工编号：" << this->m_Id
         << "\t职工姓名：" << this->m_Name
         << "\t岗位：" << this->getDeptName()
         << "\t岗位职责：完成老板交付的任务，并下发任务给员工" << endl;
}

// 获取岗位名称
string Manager::getDeptName()
{
    return string("经理");
}

// 构造函数
Manager::Manager(int id, string name, int dId)
{
	this->m_Id = id;
	this->m_Name = name;
	this->m_DeptId = dId;
}