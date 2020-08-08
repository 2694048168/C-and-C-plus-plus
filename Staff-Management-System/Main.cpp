/* file_name: the_Staff_Managerment_System.cpp
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 https://weili_yzzcq.gitee.io/, All Rights Reserved.
** Copyright © 2020 https://2694048168.github.io/, All Rights Reserved.
*/

// the area of Header files
#include <iostream>
#include "Worker_Manager.h"

// // TODO test
// #include "Worker.h"
// #include "Employee.h"
// #include "Manager.h"
// #include "Boss.h"

using namespace std;

/* 职工管理系统——the Staff Management System
** 
** 职工管理系统用于管理公司内所有员工的信息
** 员工分为三类：普通员工、经理、老板，显示信息时，有职工编号、姓名、岗位以及职责
** 普通员工职责：完成经理下发的任务
** 经理职责：完成老板的任务，并下发给员工
** 老板职责：管理公司所有事务
**
** 管理系统实现功能
** 退出管理系统：退出当前管理系统
** 增加职工信息：将信息录入文件：职工编号、姓名、部门编号
** 显示职工信息：显示公司内部所有职工信息
** 删除离职职工：按照编号删除指定的职工
** 修改职工信息：按照编号修改职工信息
** 查找职工信息：按照职工的编号或者姓名进行查找相关人员信息
** 按照编号排序：按照职工编号，进行排序，排序规则有用户指定
** 清空所有文档：清空文件记录的所有职工信息（清空之前需要再次确认，防止误删！）
**
*/
int main(int argc, char const *argv[])
{
	// // TODO test 多态
	// Worker * worker = NULL;
	// worker = new Employee(1, "张三", 1);
	// worker->showInfo();
	// delete worker;

	// Worker * worker = NULL;
	// worker = new Manager(2, "李四", 2);
	// worker->showInfo();
	// delete worker;

	// Worker * worker = NULL;
	// worker = new Boss(3, "王二", 3);
	// worker->showInfo();
	// delete worker;

	// 实例化管理者对象
	Worker_Manager wm;

	// 代表用户选择的命令
	char choice = 0;

	while (true)
	{
		// 调用显示菜单成员函数
	    wm.Show_Menu();

	    cout << "请输入您选择的菜单命令： " << endl;
	    cin >> choice;

	    switch (choice)
	    {
	    case 0: // 退出系统
	        wm.Exit_System();
	        break;

	    case 1: // 增加职工
	        wm.Add_Emp();
	        break;

	    case 2: // 显示职工
	        wm.Show_Emp();
	        break;

	    case 3: // 删除职工
	        // {
	        // 	// TODO test
	        // 	int id;
	        // 	cout << "请输入要查询职工的ID编号：" << endl;
	        // 	cin >> id;
	        // 	int ret = wm.IsExist(id);
	        // 	if (ret != -1)
	        // 	{
	        // 		cout << "职工存在，并编号为：" ret <<endl;
	        // 	}
	        // 	else
	        // 	{
	        // 		cout << "该职工不存在！" <<endl;
	        // 	}

	        // 	break;
	        // }
	        wm.Del_Emp();
	        break;

	    case 4: // 修改职工
	        wm.Mod_Emp();
	        break;

	    case 5: // 查找职工
	        wm.Find_Emp();
	        break;

	    case 6: // 排序职工
	        wm.Sort_Emp();
	        break;

	    case 7: // 清空文件
	        wm.Clean_File();
	        break;

	    default:
	        system("cls");
	    	break;
	    }
	}
	    
	system("pause");
	return 0;
}