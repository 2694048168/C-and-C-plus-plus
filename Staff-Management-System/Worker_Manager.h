/* file_name: Worker_Manager.h
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

#ifndef WORKER_MANAGER_H
#define WORKER_MANAGER_H

#include <iostream>
// using namespace std; 头文件中最好不要使用这一句，直接使用作用域符号
#include "Worker.h"
#include "Employee.h"
#include "Manager.h"
#include "Boss.h"

#include <fstream>
#define FILENAME "empFile.txt"

/* 管理类负责内容
** 1、与用户的沟通菜单界面
** 2、对职工增删改查的操作
** 3、与文件的读写交互
*/
class Worker_Manager
{
public:
	// 构造函数
	Worker_Manager();

	// 显示菜单
	void Show_Menu();

	// 退出系统
	void Exit_System();

	// 记录职工人数
	int m_EmpNum;

	// 职工数组指针
	Worker ** m_EmpArray;

	// 添加职工
	void Add_Emp();

	// 保存文件
	void save();

	// 判断文件是否为空 标志
	bool m_FileIsEmpty;

	// 统计文件中人数
	int get_EmpNum();

	// 初始化员工
	void init_Emp();

	// 显示职工
	void Show_Emp();

    // 判断员工是否存在
    int IsExist(int id);

    // 删除员工
	void Del_Emp();

	// 修改员工信息
	void Mod_Emp();

	// 查找职工
	void Find_Emp();

	// 排序职工
	void Sort_Emp();

	// 清空文件
	void Clean_File();

	// 析构函数
	~Worker_Manager();
	
};

#endif    // WORKER_MANAGER_H
