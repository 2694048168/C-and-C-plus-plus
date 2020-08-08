/*
* Copyright (c) https://2694048168.github.io
* FileName main.c
* Author 黎为
* Description 主函数作为程序的入口
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "main.h"

// 学生节点
typedef struct _STU
{
	char arrStuNum[10];
	char arrStuName[10];
	int iStuScore;
	struct _STU *pNext;
}STUNODE;
// 声明链表的头和尾
STUNODE *g_pHead = NULL;
STUNODE *g_pEnd = NULL;

// TODO 添加学生信息
void AddStuMSG(char *arrStuNum, char *arrStuName, int iStuScore);

// 清空链表
void FreeLinkData();

// 打印数据
void ShowStuData();

// 查看显示指令
void ShowOrder();

// 在链表头添加一个节点
void AddStuMSGToLinkHead(char *arrStuNum, char *arrStuName, int iStuScore);

// 查找指定学生
STUNODE *FindStuByNum(char *arrStuNum);

// 指定位置插入节点
void InsertNode(STUNODE *pTemp,char *arrStuNum, char *arrStuName, int iStuScore);

// 删除指定节点
void DeleteStuNode(STUNODE *pNode);

// 保存信息到文件
void SaveStuToFile();

// 读取文件中的学生信息
void ReadStuFromFile();


int main(int argc, char *argv[])
{
	int iOrder = -1;
	int iFlag = 1;
	char arrStuNum[10] = {"\0"};
	char arrStuName[10] = {"\0"};
	int iStuScore = 0;
    STUNODE *pTemp = NULL;//labe case 111

    // TODO 查看系统指令
    ShowOrder();

	// 读取文件中的学生信息
    ReadStuFromFile();

	while(iFlag)
	{
		//printf("Enter the Order(【9】查看系统所有指令):");
		printf("请输入指令(【9】查看系统所有指令):");
	    scanf("%d",&iOrder);
		switch (iOrder)
		{
		case 1:
			// TODO 添加学生信息
			//printf("Enter the student number:");
			printf("请输入学生的学号:");
			scanf("%s",arrStuNum);
			printf("请输入学生的姓名:");
			scanf("%s",arrStuName);
			printf("请输入学生的总学分:");
			scanf("%d",&iStuScore);

			AddStuMSG(arrStuNum, arrStuName, iStuScore);
			break;
		case 11:
			// TODO 头添加法
			/*printf("Enter the student information:\n");
		    scanf("学号:%s,姓名:%s,学分:%d",arrStuNum,arrStuName,&iStuScore);*/
		    printf("请输入学生的学号:");
			scanf("%s",arrStuNum);
			printf("请输入学生的姓名:");
			scanf("%s",arrStuName);
			printf("请输入学生的总学分:");
			scanf("%d",&iStuScore);
            AddStuMSGToLinkHead(arrStuNum, arrStuName, iStuScore);
			break;
		case 111:
		    //TODO 在指定位置节点的后面增加学生的信息
			printf("请输入指定学生学号:");
			scanf("%s",arrStuNum);
			pTemp = FindStuByNum(arrStuNum);
			if (NULL != pTemp)
			{
				// TODO 插入
				printf("请输入学生的学号:");
				scanf("%s",arrStuNum);
				printf("请输入学生的姓名:");
				scanf("%s",arrStuName);
				printf("请输入学生的总学分:");
				scanf("%d",&iStuScore);

				InsertNode(pTemp,arrStuNum,arrStuName,iStuScore);
			}
			break;
		case 2:
			// TODO 查找指定学生的信息
			printf("请输入指定学生学号:");
			scanf("%s",arrStuNum);
			// 查找
			pTemp = FindStuByNum(arrStuNum);
			// 打印
			if (NULL != pTemp)
			{
                printf("学号：%s，姓名：%s，学分：%d\n",pTemp->arrStuNum,pTemp->arrStuName,pTemp->iStuScore);
			}
			break;
		case 3:
			// TODO 修改指定学生的信息
            printf("请输入指定学生学号:");
			scanf("%s",arrStuNum);
			pTemp = FindStuByNum(arrStuNum);
			// 修改
			if (NULL != pTemp)
			{
				// TODO 智能提示是否需要修改

				// 修改学号
				printf("请输入新的学号");
                scanf("%s",arrStuNum);
				strcpy(pTemp->arrStuNum,arrStuNum);

				// 修改姓名
				printf("请输入新的姓名");
                scanf("%s",arrStuName);
				strcpy(pTemp->arrStuName,arrStuName);

				// 修改学分
				printf("请输入新的学分");
                scanf("%d",&iStuScore);
				pTemp->iStuScore = iStuScore;
			}
			break;
		case 4:
			// TODO 保存学生的信息到文件
            SaveStuToFile();
			break;
		case 5:
			// TODO 读取文件中的学生信息
            ReadStuFromFile();
			break;
		case 6:
			// TODO 删除指定学生的信息
            printf("请输入指定学生学号:");
			scanf("%s",arrStuNum);
			pTemp = FindStuByNum(arrStuNum);
			// 删除
			if (NULL != pTemp)
			{
				// TODO 调用删除节点函数
				DeleteStuNode(pTemp);
			}
			break;
		case 7:
			// TODO 恢复删除的学生的信息
			// TODO 恢复删除的学生的信息
/* 第一种方法：在内存中保存删除之前备份保存上一个链表，占据内存资源大，读取速度快
第二种方法：在文件中备份每一次删除的链表，读取速度慢
第三种方法：在文件中进行一次备份，恢复到数据最初状态 */
            // 释放链表
			FreeLinkData();
			g_pHead = NULL;
			g_pEnd = NULL;
			// 添加节点
			ReadStuFromFile();
			printf("学生信息已经恢复到最初状态！\n");
			break;
		case 8:
			// TODO 显示所有学生的信息
			ShowStuData();
			break;
		case 9:
			// TODO 查看系统指令
			ShowOrder();
			break;
		case 0:
			// TODO 退出学生管理系统
			iFlag = 0;
			break;
		default:
			printf("Please Enter the Right Order[0-8]!\n");
			break;
		}
	}

    // 保存信息到文件
	SaveStuToFile();
	// 清空链表
    FreeLinkData();

	system("pause");
	return 0;
}


// TODO 添加学生信息 【尾添加法】
void AddStuMSG(char *arrStuNum, char *arrStuName, int iStuScore)
{
	// 第一步，检验参数的合法性
	if(NULL == arrStuName || NULL == arrStuNum || iStuScore < 0)
	{
		printf("学生录入的信息出错！\n");
		return ;
	}

	// 第二步，处理业务逻辑
	// 创建新节点
	STUNODE *pTemp = malloc (sizeof (STUNODE));
	// 节点成员符初始值
	strcpy(pTemp->arrStuNum,arrStuNum);
	strcpy(pTemp->arrStuName,arrStuName);
	pTemp->iStuScore = iStuScore;
	pTemp->pNext = NULL;

	// 连接节点成为链
	if(NULL == g_pEnd || NULL == g_pHead)
	{
		g_pHead = pTemp;
		// g_pEnd = pTemp;
	}
	else
	{
		g_pEnd->pNext = pTemp;
		// g_pEnd = pTemp;
	}
	g_pEnd = pTemp;
}

// 清空链表
void FreeLinkData()
{
	STUNODE *pTemp = g_pHead;

	while(g_pHead != NULL)
	{
		// 保存需要删除的节点
		pTemp = g_pHead;
		// 头结点指针指向下一个节点
		g_pHead->pNext = g_pHead;
		// 释放节点内存资源
		free(pTemp);
	}
}

// 打印数据
void ShowStuData()
{
	STUNODE *pTemp = g_pHead;
	// pTemp->pNext = NULL;

    while (pTemp != NULL)
	{
		printf ("学号:%s, 姓名:%s, 学分:%d\n",pTemp->arrStuNum,pTemp->arrStuName,pTemp->iStuScore);
		// 向下移动节点
		pTemp = pTemp->pNext;
	}
}

// 查看显示指令
void ShowOrder()
{
	printf("******************学生信息管理系统******************\n");
    printf("****************本系统的操作指令如下****************\n");
    printf("***          1、 增加学生的信息(尾结点法)        ***\n");
    printf("***         11、 增加学生的信息(头结点法)        ***\n");
    printf("***        111、 在指定位置增加学生的信息        ***\n");
    printf("***          2、 查找指定学生的信息(学号)        ***\n");
    printf("***          3、 修改指定学生的信息              ***\n");
    printf("***          4、 保存学生的信息到文件            ***\n");
    printf("***          5、 读取文件中的学生信息            ***\n");
    printf("***          6、 删除指定学生的信息              ***\n");
    printf("***          7、 恢复删除的学生的信息            ***\n");
    printf("***          8、 显示所有学生的信息              ***\n");
    printf("***          9、 查看系统所有指令                ***\n");
    printf("***          0、 退出学生管理系统                ***\n");
    printf("****************************************************\n");
}

// 在链表头添加一个节点,【头添加法】
void AddStuMSGToLinkHead(char *arrStuNum, char *arrStuName, int iStuScore)
{
	// 第一步，检验参数的合法性
	if(NULL == arrStuName || NULL == arrStuNum || iStuScore < 0)
	{
		printf("学生录入的信息出错！\n");
		return ;
	}

	// 创建一个节点
	STUNODE *pTemp = malloc (sizeof (STUNODE));
	// 节点成员符初始值
	strcpy(pTemp->arrStuNum,arrStuNum);
	strcpy(pTemp->arrStuName,arrStuName);
	pTemp->iStuScore = iStuScore;
	pTemp->pNext = NULL;

	// 链表为空
	if (NULL == g_pHead || NULL == g_pEnd)
	{
		g_pHead = pTemp;
		// g_pEnd = pTemp;
	}
	else
	{
		// 新节点的下一个节点 指向头结点
		pTemp->pNext = g_pHead;
		// 头结点向前移动
		// g_pHead = pTemp;
	}
	g_pHead = pTemp;
}

// 查找指定学生
STUNODE *FindStuByNum(char *arrStuNum)
{
	// 检测参数合法性
	if (NULL == arrStuNum)//赋值语句和相等判断，最佳写法
	{
		printf("学号输入错误！\n");
		return NULL;
	}

	// 判断链表是否为空
	if (NULL == g_pHead || NULL == g_pEnd)
	{
		printf("当前链表为空[NULL]！\n");
		return NULL;
	}
	//遍历链表
	STUNODE *pTemp = g_pHead;
    while(pTemp != NULL)
	{
		if (0 == strcmp(pTemp->arrStuNum,arrStuNum))
		{
			return pTemp;
		}
		// 向下移动节点
		pTemp = pTemp->pNext;
	}
	printf("查询无果，查无节点");
	return NULL;
}

// 指定位置插入节点
void InsertNode(STUNODE *pTemp,char *arrStuNum, char *arrStuName, int iStuScore)
{
	// 创建节点
	STUNODE *pNewTemp = malloc(sizeof(STUNODE));

	// 节点成员符初始值
	strcpy(pNewTemp->arrStuNum,arrStuNum);
	strcpy(pNewTemp->arrStuName,arrStuName);
	pNewTemp->iStuScore = iStuScore;
	pNewTemp->pNext = NULL;// 记得一定赋值为空NULL

	// 插入节点
    if (pTemp == g_pEnd)// 尾节点
	{
		g_pEnd->pNext = pNewTemp;
		g_pEnd = pNewTemp;
	}
	else
	{
        pNewTemp->pNext = pTemp->pNext;
        pTemp->pNext = pNewTemp;
	}
}

// 删除指定节点
void DeleteStuNode(STUNODE *pNode)
{
	// 只有一个节点
	if (g_pEnd == g_pHead)
	{
		free(g_pEnd);
		g_pEnd = NULL;
		g_pHead = NULL;
	}
	// 只有两个节点
	else if (g_pHead->pNext == g_pEnd)
	{
		if (g_pHead == pNode)
		{
			free(g_pHead);
			g_pHead = g_pEnd;
		}
		else
		{
			free(g_pEnd);
			g_pEnd = g_pHead;
			g_pHead->pNext = NULL;
		}
	}
	// 一般情况，三个节点以上
	else
	{
		STUNODE *pTemp = g_pHead;
		if (g_pHead == pNode)// 要删除的是头节点
		{
			// 记住头
			pTemp == g_pHead;
			// 删除头
			g_pHead = g_pHead->pNext;
			free(pTemp);
			pTemp = NULL;

			return ;
		}
		else
		{
		    while(pTemp != NULL)
		    {
			    if (pTemp->pNext == pNode)
			    {
				    // 删除
					if (g_pEnd == pNode)//要删除的是尾节点
					{
						free(pNode);
						pNode = NULL;
						g_pEnd = pTemp;
						g_pEnd->pNext = NULL;
						return ;
					}
					else// 要删除的是中间节点
					{
						// 记住要删除的节点
						STUNODE *p = pTemp->pNext;
						// 链接下一个节点
						pTemp->pNext = pTemp->pNext->pNext;
						// 释放节点P
						free(p);
						p = NULL;// 指针释放之后一定赋值为空，防止野指针
						return ;
					}
			    }
			    pTemp = pTemp->pNext;
		    }
		}
	}
}

// 保存信息到文件
void SaveStuToFile()
{
	FILE *pFile = NULL;
	STUNODE *pTemp = g_pHead;
	//static char *strBuff[30] = {"\0"};
	char *strBuff[30] = {"\0"};
	//static char *strScroe[10] ={"\0"};
	char *strScroe[10] ={"\0"};

	// 判断链表是否为空
	if (NULL == g_pHead)
	{
		printf("链表为空，没有学生");
		return ;
	}

	// 打开文件
	pFile = fopen("./stu.txt","wb+");
	//pFile = fopen("./stu.txt","w+");
	if (NULL == pFile)
	{
		printf("文件打开失败！\n");
		return ;
	}
	// 操作文件
	while(pTemp != NULL)
	{
		// 学号赋值到strBuff
		// TODO cannot access memory at address
	    // strcat函数不能访问，堆栈溢出，？？？？
		strcpy(strBuff,pTemp->arrStuNum);
		strcat(strBuff,",");// 以,间隔每一个字段,是一个字符串用双引号“”
		// 姓名
		strcat(strBuff,pTemp->arrStuName);
		strcat(strBuff,",");
		// 学分
		itoa(pTemp->iStuScore,strScroe,10);
		strcat(strBuff,strScroe);

		fwrite(strBuff,1,strlen(strBuff),pFile);
		fwrite("\r\n",1,strlen("\r\n"),pFile);// 每一个条记录进行换行储存

		pTemp = pTemp->pNext;
	}
	// 关闭文件
	fclose(pFile);
}

// 读取文件中的学生信息
void ReadStuFromFile()
{
    //char strBuff[30] = {"\0"};
    char strBuff[30] = {0};
	char strStuNum[10] = {0};
	char strStuName[10] = {0};
	char strStuScore[10] = {0};
	//int iCount = 0;// 读取文件的字段间隔标志位
	//int j = 0;

	// 打开文件
	FILE *pFile = fopen("./stu.txt","ab+");
	//FILE *pFile = fopen("./stu.txt","r+");
	if (NULL == pFile)
	{
		printf("文件打开失败！\n");
		return ;
	}
	// 操作指针，读取函数
	//while(EOF != fgets(strBuff,30,pFile))// EOF 文件结束标识符
	while(NULL != fgets(strBuff,30,pFile))// EOF 文件结束标识符
	{
		int i = 0;
		int iCount = 0;
		int j = 0;
		int n = 0;

		for (i = 0; strBuff[i] != '\r'; i++)
		{
			if (0 == iCount)// 没到分隔符 ,
			{
				strStuNum[i] = strBuff[i];
				if (',' == strBuff[i])
				{
					strStuNum[i] = '\0';// 去掉分隔符 ,
					iCount++;
				}
			}
			else if (1 == iCount)// 第一个分隔符 ,
			{
				strStuName[j] = strBuff[i];
				//j++;
				if (',' == strBuff[i])
				{
					strStuName[j] = '\0';// 去掉分隔符 ,
					iCount++;
					//j = 0;
				}
				j++;
			}
			else // 第二个分隔符 ,
			{
				strStuScore[n] = strBuff[i];
				n++;
			}
		}
		//int iStuScore = atoi(strStuScore);
		// 插入节点到链表
		// TODO 添加学生信息
        AddStuMSG(strStuNum, strStuName, atoi(strStuScore));
	}
	// 关闭文件
	fclose(pFile);
}
