/* file_name: Main.cpp
** author: Wei Li
** date: 2020-06-16
** Copyright © 2020 Wei Li, All Rights Reserved.
** GitHub: https://github.com/2694048168/
** Gitee：https://gitee.com/weili_yzzcq/
** Blog：https://2694048168.github.io/
** Blog：https://weili_yzzcq.gitee.io/ 
** Blog：https://blog.csdn.net/weixin_46782218/
*/

// 头文件区
// the area of Header files
#include <iostream>
#include <string>
using namespace std;

// 全局变量区
// the area of Global variables
// 通讯录最大储存联系人
#define MAX 500

// 联系人结构体
struct Person
{
	// 姓名
    string m_Name;
	// 性别
    unsigned int m_Sex;
	// 年龄
    unsigned int m_Age;
	// 电话
    string m_Phone;
	// 地址
    string m_Addr;
};

// 通讯录结构体
struct Addressbooks
{
	// 通讯录中保存联系人数组
	struct Person personArray[MAX];

	// 通讯录中当前记录的联系人个数
	unsigned int m_Size;
};

// 全局函数声明区
// the area of Global function declaration
/* 功能：显示菜单命令
** 返回值：void
** 函数名：showMenu
** 参数列表：void
** 参数传递类型：void
** 函数调用：showMenu();
*/
void showMenu();

/* 功能：添加联系人
** 返回值：void
** 函数名：addPerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：addPerson(&abs);
*/
void addPerson(Addressbooks * abs);

/* 功能：显示联系人
** 返回值：void
** 函数名：showPerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：showPerson(&abs);
*/
void showPerson(Addressbooks * abs);

/* 功能：检测联系人是否在当前通讯录中
** 返回值：int
** 函数名：isExist
** 参数列表：通讯录结构体abs、检测联系人姓名name
** 参数传递类型：地址传递、值传递
** 函数调用：isExist(&abs);
*/
int isExist(Addressbooks * abs, string name);

/* 功能：删除联系人
** 返回值：void
** 返回值说明：i代表着该联系人在通讯录中的索引，-1代表着查无此联系人
** 函数名：deletePerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：deletePerson(abs, name);
*/
void deletePerson(Addressbooks * abs);

/* 功能：查找联系人
** 返回值：void
** 函数名：findPerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：findPerson(&abs);
*/
void findPerson(Addressbooks * abs);

/* 功能：修改联系人
** 返回值：void
** 函数名：modifyPerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：modifyPerson(&abs);
*/
void modifyPerson(Addressbooks * abs);

/* 功能：删除联系人
** 返回值：void
** 函数名：cleanPerson
** 参数列表：通讯录结构体abs
** 参数传递类型：地址传递
** 函数调用：cleanPerson(&abs);
*/
void cleanPerson(Addressbooks * abs);


/* 通讯录管理系统——the Contacts Management System
** 
** 添加联系人：向通讯录添加新人（姓名、性别、年龄、联系电话、家庭住址）最多记录500人
** 显示联系人：显示通讯录中所有联系人信息
** 删除联系人：按姓名索引删除指定联系人
** 查找联系人：按姓名索引查找指定联系人
** 修改联系人：按姓名索引修改指定联系人
** 清空联系人：清空通讯录中所有的联系人信息
** 退出通讯录：退出当前使用的通讯录系统 
**
*/
int main(int argc, char const *argv[])
{
	// 创建通讯录结构体变量
	Addressbooks abs;
	// 初始化通讯录当前联系人个数
	abs.m_Size = 0;

	// 接受用户输入命令
	unsigned int select = 0;

	// 循环接受用户命令
	while(true)
	{
		// 菜单函数
	    showMenu();

	    // 接受用户输入
		cin >> select;

		switch(select)
		{
		case 1: // 1、添加联系人
		    addPerson(&abs);
			break;

		case 2: // 2、显示联系人
		    showPerson(&abs);
			break;

		case 3: // 3、删除联系人
            deletePerson(&abs);
			break;

		case 4: // 4、查找联系人
		    findPerson(&abs);
			break;

		case 5: // 5、修改联系人
		    modifyPerson(&abs);
			break;

		case 6: // 6、清空联系人
		    cleanPerson(&abs);
			break;

		case 0: // 0、退出通信录
		    cout << "欢迎下次使用系统！" << endl;
		    system("pause");
		    return 0;
			break;

		default:
		    break;
		}
	}
    
	return 0;
}


// 全局函数实现区
// the area of Global function implementation
/* Function：show the user menu command
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：deletePerson();
*/
void showMenu()
{
    cout << "=============================================" << endl;
    cout << "============== 1、添加联系人 ============== " << endl;
    cout << "============== 2、显示联系人 ============== " << endl;
    cout << "============== 3、删除联系人 ============== " << endl;
    cout << "============== 4、查找联系人 ============== " << endl;
    cout << "============== 5、修改联系人 ============== " << endl;
    cout << "============== 6、清空联系人 ============== " << endl;
    cout << "============== 0、退出通讯录 ============== " << endl;
    cout << "=============================================" << endl;
}

/* Function：add the person to Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：addPerson(&abs);
*/
void addPerson(Addressbooks * abs)
{
	// 先判断通讯录是否满了，检测MAX
	if (abs->m_Size == MAX)
	{
		cout << "当前通讯录已满，无法添加！" << endl;
		return;
	}
	else
	{
		// 添加联系人
		cout << "请输入姓名" << endl;
		string name;
		cin >> name;
		abs->personArray[abs->m_Size].m_Name = name;

		cout << "请输入性别" << endl;
		cout << "1 -- 男" << endl;
		cout << "2 -- 女" << endl;
		unsigned int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
		    {
                abs->personArray[abs->m_Size].m_Sex = sex;
                break;
		    }
		    cout << "输入有误，请重新输入！" << endl;
		}

		cout << "请输入年龄" << endl;
		unsigned int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 150)
		    {
                abs->personArray[abs->m_Size].m_Age = age;
                break;
		    }
		    cout << "输入有误，请重新输入！" << endl;
		}

		cout << "请输入联系电话" << endl;
		string phone;
		cin >> phone;
		abs->personArray[abs->m_Size].m_Phone = phone;

		cout << "请输入地址" << endl;
		string address;
		cin >> address;
		abs->personArray[abs->m_Size].m_Addr = address;

		// 更新当前通讯录人数
		abs->m_Size++;

		cout << "添加联系人成功！" << endl;
		system("pause");
		system("cls");
	}
}

/* Function：show the persons of Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：showPerson(&abs);
*/
void showPerson(Addressbooks * abs)
{
	// 判断通讯录是否为空
	if (abs->m_Size == 0)
	{
		cout << "当前通讯录为空！" << endl;
	}
	else
	{
		for (unsigned int i =0; i < abs->m_Size; i++)
		{
			cout << "姓名： " << abs->personArray[i].m_Name << "\t";
			cout << "性别： " << ( abs->personArray[i].m_Sex == 1 ?"男":"女" ) << "\t";
			cout << "年龄： " << abs->personArray[i].m_Age << "\t";
			cout << "电话： " << abs->personArray[i].m_Phone << "\t";
			cout << "地址： " << abs->personArray[i].m_Addr << endl;
		}
	}

	system("pause");
	system("cls");
}

/* Function：Detect this contact is or not in the Address Book
** Return：int
** Return Value: the i is the person of index in the Address Book, and the -1 means disdetect the person 
** Parameter_one：abs
** Parameter_two：name
** Parameter Passing Type：Value Passing
** Function Call：isExist(abs, name);
*/
int isExist(Addressbooks * abs, string name)
{
	for (unsigned int i = 0 ; i < abs->m_Size; i++)
	{
		if (abs->personArray[i].m_Name == name)
		{
			return i;
		}
	}

	return -1;
}

/* Function：Delete the person from the Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：deletePerson(&abs);
*/
void deletePerson(Addressbooks * abs)
{
	cout << "请输入删除联系人的姓名" << endl;
    string name;
    cin >> name;

    int ret = isExist(abs,name);
	if (ret != -1)
	{
		for (unsigned int i = ret; i < abs->m_Size; i++)
		{
			// 数据前移
			abs->personArray[i] = abs->personArray[i + 1];
		}
		abs->m_Size--;
		cout << "删除联系人成功！" << endl;
	}
    else
    {
        cout << "当前通讯录查无此人！" << endl;
    }

    system("pause");
	system("cls");
}

/* Function：find the persons is or not in the Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：findPerson(&abs);
*/
void findPerson(Addressbooks * abs)
{
	cout << "请输入查找联系人的姓名" << endl;
    string name;
    cin >> name;

    int ret = isExist(abs,name);
	if (ret != -1)
	{
		cout << "姓名： " << abs->personArray[ret].m_Name << "\t";
		cout << "性别： " << ( abs->personArray[ret].m_Sex == 1 ?"男":"女" ) << "\t";
		cout << "年龄： " << abs->personArray[ret].m_Age << "\t";
		cout << "电话： " << abs->personArray[ret].m_Phone << "\t";
		cout << "地址： " << abs->personArray[ret].m_Addr << endl;
	}
    else
    {
        cout << "当前通讯录查无此人！" << endl;
    }

    system("pause");
	system("cls");
}

/* Function：modify the person in the  Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：modifyPerson(&abs);
*/
void modifyPerson(Addressbooks * abs)
{
    cout << "请输入修改联系人的姓名" << endl;
    string name;
    cin >> name;

    int ret = isExist(abs,name);
	if (ret != -1)
	{
		// 修改联系人
		cout << "请输入姓名" << endl;
		string name;
		cin >> name;
		abs->personArray[ret].m_Name = name;

		cout << "请输入性别" << endl;
		cout << "1 -- 男" << endl;
		cout << "2 -- 女" << endl;
		unsigned int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
		    {
                abs->personArray[ret].m_Sex = sex;
                break;
		    }
		    cout << "输入有误，请重新输入！" << endl;
		}

		cout << "请输入年龄" << endl;
		unsigned int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 150)
		    {
                abs->personArray[ret].m_Age = age;
                break;
		    }
		    cout << "输入有误，请重新输入！" << endl;
		}

		cout << "请输入联系电话" << endl;
		string phone;
		cin >> phone;
		abs->personArray[ret].m_Phone = phone;

		cout << "请输入地址" << endl;
		string address;
		cin >> address;
		abs->personArray[ret].m_Addr = address;
	}
    elsem
    {
        cout << "当前通讯录查无此人！" << endl;
    }

    system("pause");
	system("cls");	
}

/* Function：clean the persons of the total Address Book
** Return：void
** Parameter：abs
** Parameter Passing Type：Address Passing or Pointer Passing
** Function Call：cleanPerson(&abs);
*/
void cleanPerson(Addressbooks * abs)
{
	// 逻辑清空
	abs->m_Size = 0;
    cout << "通讯录联系人已清空！" << endl;
    system("pause");
	system("cls");		
}
