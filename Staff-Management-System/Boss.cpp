/* file_name: Boss.cpp
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

#include <iostream>
#include "Boss.h"
#include <string>
using namespace std;	

// 显示所有信息
void Boss::showInfo()
{
    cout << "职工编号：" << this->m_Id
         << "\t职工姓名：" << this->m_Name
         << "\t岗位：" << this->getDeptName()
         << "\t岗位职责：管理公司所有事务" << endl;
}

// 获取岗位名称
string Boss::getDeptName()
{
    return string("老板");
}

// 构造函数
Boss::Boss(int id, string name, int dId)
{
	this->m_Id = id;
	this->m_Name = name;
	this->m_DeptId = dId;
}