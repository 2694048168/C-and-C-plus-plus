/*
* Copyright (c) https://2694048168.github.io
* FileName main.c
* Author 黎为
* Description 主函数作为程序的入口
*/

/*
* 用C语言实现Win32程序，完成俄罗斯方块游戏程序
* 1、窗口创建
*    初始化窗口类
*    注册、创建、显示窗口
*    消息循环、回调函数
* 
* 2、游戏业务逻辑实现
*    二维数组布局
*
* 3、游戏结束
*
* 4、音乐媒体播放器
*/
#include <windows.h>
#include <time.h>
#include <mmsystem.h>
#pragma comment(lib,"winmm.lib")

#define DEF_TIMER1 1234 // 定时器

int g_iSqareID = -1;
int g_iLine = -1;
int g_iList = -1;
int g_iScore = 0;

// 回调函数
LRESULT CALLBACK PELouSi(HWND hWnd, UINT nMsg, WPARAM wParam, LPARAM lParam);

// 绘制函数
void OnPaint(HDC hDc);

// 初始化数据
void onCreate();

// 背景数组
char g_arrBackGroud[20][10] = {0};
// 随机方块
char g_arrSqare[2][4] = {0};

// 显示方块
void PaintSqare(HDC hMemDC);

// 随机方块产生
int CreateRandomSqare();

// 随机方块贴到背景
void CopySqareToBack();

// Enter回车按键处理函数
void OnReturn(HWND hWnd);

// 方块下落
void SqareDown();

// 定时器响应函数
void OnTime(HWND hWnd);

// 方块停止在最底部 0不可以落，1可以落
int CanSqareDown();
// 方块停止在最底部,不覆盖其他方块，而是重叠 0不可以落，1可以落
int CanSqareDown2();
// 方块是否能够左移，边界判断,0不可以左移，1可以左移
int CanSqareLeft();
// 方块是否能够左移，边界左有方块情况判断,0不可以左移，1可以左移
int CanSqareLeft2();
// 方块是否能够右移，边界判断,0不可以左移，1可以左移
int CanSqareRight();
// 方块是否能够右移，边界左有方块情况判断,0不可以左移，1可以左移
int CanSqareRight2();

// 1 to 2 停止标志
void Change1To2();

// 显示2的情况方块，停止时
void ShowSqare2(HDC hMemDC);

// 左移动消息
void OnLeft(HWND hWnd);
// 右移动消息
void OnRight(HWND hWnd);

// 方块左移
void SqareLeft();
// 方块右移
void SqareRight();

// 向下加速消息
void OnDown(HWND hWnd);

// 向上变型
void OnChangeSqare(HWND hWnd);

// 一般形状变型
void ChangeSqare();
// 一般形状方块能否变型条件判断
int CanSqareChangeShape();
// 长方形变型
void ChangeLineSqare();
// 长方形方块能否变型条件判断
int CanLineSqareChangeShape();

// 消除一行方块
void DestroyOneLineSqare();

// 显示积分
void ShowScore(HDC hMemDC);

// TODO 游戏结束失败
int CanGameOver();

// WINAPI 调用约定 _stdcall 参数的入栈顺序，栈空间的清理者
// 参数1，句柄：一个数，是窗口的唯一标识
// 参数2，前一个句柄：同一个窗口同时打开多个
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR IpCmdLine, int nCmdShow)
{
	// 初始化窗口类
	WNDCLASSEX wc;  // WNDCLASS
	HWND hWnd;  // 窗口句柄
	MSG msg;  // 消息结构体

	wc.cbClsExtra = 0;
	wc.cbSize = sizeof(WNDCLASSEX);
	wc.cbClsExtra = 0;
	wc.hbrBackground = (HBRUSH)COLOR_INFOTEXT; // 背景颜色
	wc.hCursor = LoadCursor(NULL,IDC_ARROW); // TODO 加载系统光标，可以自定义光标图片
	//wc.hIcon = LoadIcon(hInstance,HAKEINTRESOURCE(IDI_ICON)); // TODO 可以自定义加载图标
	wc.hIcon = LoadIcon(NULL,IDI_INFORMATION); // TODO 加载系统图标，可以自定义加载图标
	wc.hIconSm = NULL; // NULL 默认与Icon一致
	wc.hInstance = hInstance;
	wc.lpfnWndProc = PELouSi; // 回调函数地址
	wc.lpszClassName = "els"; //窗口类名字，操作系统看的
    wc.lpszMenuName = NULL;
    wc.style = CS_HREDRAW | CS_VREDRAW; // 窗口风格

    // 注册窗口
    if (0 == RegisterClassEx(&wc))
    {
        // 注册出错
        //int a = GetLastError();
        //printf("窗口注册出错，出错码为：%d",a);
        return 0;
    }

    // 创建窗口
    hWnd = CreateWindowEx(WS_EX_TOPMOST,"els","俄罗斯方块",WS_OVERLAPPEDWINDOW,100,100,500,650,NULL,NULL,hInstance,NULL);
    if (NULL == hWnd)// 窗口句柄
    {
        return 0;
    }

    // 显示窗口
    ShowWindow(hWnd,SW_SHOWNORMAL);// nCmdShow

    // 播放音乐
	// 需要在编译器中自己连接库文件libwinmm.a，是编译器自带的静态库文件，在链接器中加入
    mciSendString((LPCSTR)"open .\\music.mp3 alias mymusic", NULL, 0, NULL);
	mciSendString((LPCSTR)"play mymusic", NULL, 0, NULL);


    // 消息循环
    while (GetMessage(&msg,NULL,0,0))// 循环获取消息队列的消息
    {
        // 翻译消息
        TranslateMessage(&msg);
        // 分发消息：标准消息、命令消息、通知消息、自定义消息
        DispatchMessage(&msg);
    }

	return 0;
}

// 回调函数
LRESULT CALLBACK PELouSi(HWND hWnd, UINT nMsg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT pt;
    HDC hDc;

    switch (nMsg)
    {
        // 回调函数第一次调用，只产生一次
    case WM_CREATE:
        // 用于数据初始化,什么都没有，编译器会优化处理
        //GetLastError(); // 调试用的，编译器不会优化
        onCreate();
        break;
    case WM_TIMER:
        // TODO Enter按键定时器产生消息
        // 定时器响应函数
        OnTime(hWnd);
        break;
        // 回调函数第二次调用，窗口更新(重新绘制)
    case WM_PAINT:
        // 比如窗口拉大缩小变化重绘
        hDc = BeginPaint(hWnd,&pt); // getDc,窗口可操作区域标识

        // 绘制开始
        OnPaint(hDc);

        EndPaint(hWnd,&pt);
        break;
    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_RETURN:
            // Enter回车按键处理函数
            OnReturn(hWnd);
            break;
        case VK_LEFT:
            // 左移动
            OnLeft(hWnd);
            break;
        case VK_RIGHT:
            // 右移动
            OnRight(hWnd);
            break;
        case VK_UP:
            // 向上变型
            OnChangeSqare(hWnd);
            break;
        case VK_DOWN:
            // 向下加速
            OnDown(hWnd);
            break;
        }
        break;
        // 退出程序窗口
    case WM_DESTROY:
        // 关闭定时器
        KillTimer(hWnd,DEF_TIMER1);
        PostQuitMessage(0); // WM_CLOSE, WM_DESTROY, WM_QUIT
        break;
    }

    return DefWindowProc(hWnd, nMsg, wParam, lParam); //
}

// 绘制函数
void OnPaint(HDC hDc)
{
    // 创建兼容性DC-ID编号
    HDC hMemDC = CreateCompatibleDC(hDc);
    // 创建一张画纸-位图
    HBITMAP hBitmapBack = CreateCompatibleBitmap(hDc,500,600);
    // 关联ID编号与画纸位图
    SelectObject(hMemDC,hBitmapBack);

    // 显示方块
    PaintSqare(hMemDC);
    ShowSqare2(hMemDC);//停止时显示方块
    // 显示积分
    ShowScore(hMemDC);

    // 传递图片 从内存dc传递到窗口
    BitBlt(hDc,0,0,500,600,hMemDC,0,0,SRCCOPY);

    // 释放兼容性DC
    DeleteObject(hBitmapBack);
    DeleteDC(hMemDC);
}

// 初始化数据
void onCreate()
{
    // 随机方块，随机种子
    srand((unsigned int)time(NULL));
    // 产生一次
    CreateRandomSqare();
    // 复制一次
    CopySqareToBack();
}

// 显示方块
void PaintSqare(HDC hMemDC)
{
    int i = 0;
    int j = 0;
    HBRUSH hOldBrush;//原来画刷
    HBRUSH hNewBrush;//新画刷

    // 大方块背景
    Rectangle(hMemDC,0,0,300,600);

    // 指定显示方块
    /*g_arrBackGroud[2][4] = 1;
    g_arrBackGroud[3][3] = 1;
    g_arrBackGroud[3][4] = 1;
    g_arrBackGroud[3][5] = 1;*/

    // 画刷颜色，给方块涂色

    hNewBrush = CreateSolidBrush(RGB(63,191,49));
    hOldBrush = (HBRUSH)SelectObject(hMemDC,hNewBrush);

    // 遍历
    for (i = 0; i < 20; i++)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                // 画小方块
                Rectangle(hMemDC,j*30,i*30,j*30+30,i*30+30);
            }
        }

    }
    // 释放画刷
    hNewBrush = (HBRUSH)SelectObject(hMemDC,hOldBrush);
    DeleteObject(hNewBrush);
}

// 随机方块产生
int CreateRandomSqare()
{
    // 七个不同形状方块
    int iInde = rand()%7;

    switch(iInde)
    {
    case 0:
        g_arrSqare[0][0] = 1,g_arrSqare[0][1] = 1,g_arrSqare[0][2] = 0,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 0,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 1,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =3;
        break;
    case 1:
        g_arrSqare[0][0] = 0,g_arrSqare[0][1] = 1,g_arrSqare[0][2] = 1,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 1,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 0,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =3;
        break;
    case 2:
        g_arrSqare[0][0] = 1,g_arrSqare[0][1] = 0,g_arrSqare[0][2] = 0,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 1,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 0,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =3;
        break;
    case 3:
        g_arrSqare[0][0] = 1,g_arrSqare[0][1] = 0,g_arrSqare[0][2] = 0,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 1,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 1,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =3;
        break;
    case 4:
        g_arrSqare[0][0] = 0,g_arrSqare[0][1] = 1,g_arrSqare[0][2] = 0,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 1,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 1,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =3;
        break;
    case 5:
        g_arrSqare[0][0] = 0,g_arrSqare[0][1] = 1,g_arrSqare[0][2] = 1,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 0,g_arrSqare[1][1] = 1,g_arrSqare[1][2] = 1,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =4;
        break;
    case 6:
        g_arrSqare[0][0] = 1,g_arrSqare[0][1] = 1,g_arrSqare[0][2] = 1,g_arrSqare[0][3] = 0;
        g_arrSqare[1][0] = 0,g_arrSqare[1][1] = 0,g_arrSqare[1][2] = 0,g_arrSqare[1][3] = 0;
        g_iLine =0;
        g_iList =4; // TODO 以中心点为轴心进行变型
        break;
    }

    g_iSqareID = iInde;

    return iInde;
}

// 随机方块贴到背景
void CopySqareToBack()
{
    int i = 0;
    int j = 0;

    for(i = 0; i < 2; i++)
    {
        for (j = 0; j < 4; j++)
        {
            g_arrBackGroud[i][j+3] = g_arrSqare[i][j];
        }
    }
}

// Enter回车按键处理函数
void OnReturn(HWND hWnd)
{
    // 打开定时器
    SetTimer(hWnd,DEF_TIMER1,500,NULL);// 1000毫秒=1秒

    // 关闭定时器
}

// 方块下落
void SqareDown()
{
    int i = 0;
    int j = 0;

    for (i = 19; i >= 0; i--)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                g_arrBackGroud[i+1][j] = g_arrBackGroud[i][j];
                g_arrBackGroud[i][j] = 0;
            }
        }
    }
}

// 定时器响应函数
void OnTime(HWND hWnd)
{
    HDC hDc = GetDC(hWnd);
    if (1 == CanSqareDown() && 1 == CanSqareDown2())
    {
        SqareDown();
        g_iLine++;
    }
    else
    {
        // 1 to 2 停止标志
        Change1To2();
        // 消除一行方块
        DestroyOneLineSqare();

        // TODO 游戏结束失败
        if (0 == CanGameOver())
        {
            // 结束程序
            KillTimer(hWnd, DEF_TIMER1);
            return ;
        }

        // 产生随机块
        CreateRandomSqare();
        // 复制到背景上
        CopySqareToBack();
    }

    // 显示方块
    //PaintSqare(hDc);
    OnPaint(hDc);// 内存DC, 兼容性DC

    ReleaseDC(hWnd,hDc);
}

// 方块停止在最底部 0不可以落，1可以落
int CanSqareDown()
{
    int i = 0;

    for (i = 0; i < 10; i++)
    {
        if (1 == g_arrBackGroud[19][i])
        {
            return 0;// 不可以下落了，到最底部了
        }
    }
    return 1;// 还可以下落
}

// 1 to 2 停止标志
void Change1To2()
{
    int i = 0,
        j = 0;
    for (i = 0; i < 20; i++)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                g_arrBackGroud[i][j] = 2;
            }
        }
    }
}

// 显示2的情况方块，停止时方块颜色改变
void ShowSqare2(HDC hMemDC)
{
    int i = 0,
        j = 0;

    HBRUSH hOldBrush;
    HBRUSH hNewBrush;
    hNewBrush = CreateSolidBrush(RGB(233,27,182));
    hOldBrush = (HBRUSH)SelectObject(hMemDC,hNewBrush);

    for (i = 0; i < 20; i++)
    {
        for (j = 0; j < 10; j++)
        {
            if (2 == g_arrBackGroud[i][j])
            {
                // 画小方块
                Rectangle(hMemDC,j*30,i*30,j*30+30,i*30+30);
            }
        }
    }

   hNewBrush = (HBRUSH)SelectObject(hMemDC,hOldBrush);
   DeleteObject(hNewBrush);
}

// 方块停止在最底部,不覆盖其他方块，而是重叠 0不可以落，1可以落
int CanSqareDown2()
{
    int i = 0,
        j = 0;
    for (i = 19; i >= 0; i--)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                 if (2 == g_arrBackGroud[i+1][j])
                 {
                     return 0;
                 }
            }
        }
    }
    return 1;
}

// 左移动
void OnLeft(HWND hWnd)
{
    // 方块左移
    if (1 == CanSqareLeft() && 1 == CanSqareLeft2())
    {
        HDC hDc = GetDC(hWnd);
        SqareLeft();
        g_iList--;
        // 显示方块
        OnPaint(hDc);

        ReleaseDC(hWnd,hDc);
    }
}

// 方块左移
void SqareLeft()
{
    int i = 0,
        j = 0;

    for (i = 0; i < 20; i++)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                 g_arrBackGroud[i][j-1] = g_arrBackGroud[i][j];
                 g_arrBackGroud[i][j] = 0;
            }
        }
    }
}

// 方块是否能够左移，边界判断,0不可以左移，1可以左移
int CanSqareLeft()
{
    int i = 0;

    for (i = 0; i < 20; i++)
    {
         if (1 == g_arrBackGroud[i][0])
         {
             return 0;
         }
    }
    return 1;
}

// 方块是否能够左移，边界左有方块情况判断,0不可以左移，1可以左移
int CanSqareLeft2()
{

    int i = 0,
        j = 0;

    for (i = 0; i < 20; i++)
    {
        for (j = 0; j < 10; j++)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                 if (2 == g_arrBackGroud[i][j-1])
                 {
                     return 0;
                 }
            }
        }
    }

    return 1;
}

// 右移动消息
void OnRight(HWND hWnd)
{
    // 方块右块
    if (1 == CanSqareRight() && 1 == CanSqareRight2())
    {
        HDC hDc = GetDC(hWnd);
        SqareRight();
        g_iList++;
        // 显示方块
        OnPaint(hDc);

        ReleaseDC(hWnd,hDc);
    }
}

// 方块右移
void SqareRight()
{
    int i = 0,
        j = 0;

    for (i = 0; i < 20; i++)
    {
        for (j = 9; j >= 0; j--)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                 g_arrBackGroud[i][j+1] = g_arrBackGroud[i][j];
                 g_arrBackGroud[i][j] = 0;
            }
        }
    }
}

// 方块是否能够右移，边界判断,0不可以左移，1可以左移
int CanSqareRight()
{
    int i = 0;

    for (i = 0; i < 20; i++)
    {
         if (1 == g_arrBackGroud[i][9])
         {
             return 0;
         }
    }

    return 1;
}

// 方块是否能够右移，边界左有方块情况判断,0不可以左移，1可以左移
int CanSqareRight2()
{
    int i = 0,
        j = 0;

    for (i = 0; i < 20; i++)
    {
        for (j = 9; j >= 0; j--)
        {
            if (1 == g_arrBackGroud[i][j])
            {
                 if (2 == g_arrBackGroud[i][j+1])
                 {
                     return 0;
                 }
            }
        }
    }

    return 1;
}

// 向下加速消息
void OnDown(HWND hWnd)
{
    OnTime(hWnd);
}

// 向上变型
void OnChangeSqare(HWND hWnd)
{
    HDC hDc = GetDC(hWnd);

    switch (g_iSqareID)
    {
    case 0:
    case 1:
    case 3:
    case 2:
    case 4:
        // 一般形状
        // 一般形状方块能否变型条件判断
        if (1 == CanSqareChangeShape())
        {
            ChangeSqare();
        }
        else
        {
            return ;
        }
        break;
    case 5:
        // 正方形，不需要变型
        return ;
    case 6:
        // 长方形变型
        // 长方形方块能否变型条件判断,0不可以变型，1可以变型
        if (1 == CanLineSqareChangeShape())
        {
             ChangeLineSqare();
        }
        break;
    }

    // 显示
    OnPaint(hDc);
    // 释放DC
    ReleaseDC(hWnd,hDc);
}

// 一般形状变型
void ChangeSqare()
{
    int i = 0;
    int j = 0;
    int iTemp = 2;

    char arrSqare[3][3] = {0};

    // 背景复制出来
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            arrSqare[i][j] = g_arrBackGroud[g_iLine + i][g_iList + j];
        }
    }

    // 变型后复制回去
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            g_arrBackGroud[g_iLine + i][g_iList + j] = arrSqare[iTemp][i];
            iTemp--;
        }
        iTemp = 2;
    }

}

// 一般形状方块能否变型条件判断 0不可以变型，1可以变型
int CanSqareChangeShape()
{
    int i = 0;
    int j = 0;

    // 背景复制出来
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            if (2 == g_arrBackGroud[g_iLine + i][g_iList + j])
            {
                return 0;
            }
        }
    }

    // 左右两边数组边界越界问题情况
    /*if (g_iList < 0 || g_iList > 9)
    {
        return 0;
    }*/
    if (g_iList < 0)
    {
        g_iList = 0;
    }
    else if (g_iList + 2 > 9)
    {
        g_iList = 7;
    }

    return 1;
}

// 长方形变型
void ChangeLineSqare()
{
    if (1 == g_arrBackGroud[g_iLine][g_iList - 1])// 长方形是横着的
    {
        // 背景清零
        g_arrBackGroud[g_iLine][g_iList - 1] = 0;
        g_arrBackGroud[g_iLine][g_iList + 1] = 0;
        g_arrBackGroud[g_iLine][g_iList + 2] = 0;

        if (2 == g_arrBackGroud[g_iLine + 1][g_iList])
        {
            // 复制数组
            g_arrBackGroud[g_iLine - 1][g_iList] = 1;
            g_arrBackGroud[g_iLine - 2][g_iList] = 1;
            g_arrBackGroud[g_iLine - 3][g_iList] = 1;

        }
        else if (2 == g_arrBackGroud[g_iLine + 2][g_iList])
        {
            // 复制数组
            g_arrBackGroud[g_iLine + 1][g_iList] = 1;
            g_arrBackGroud[g_iLine - 1][g_iList] = 1;
            g_arrBackGroud[g_iLine - 2][g_iList] = 1;
        }
        else
        {
             // 元素复制
            g_arrBackGroud[g_iLine - 1][g_iList] = 1;
            g_arrBackGroud[g_iLine + 1][g_iList] = 1;
            g_arrBackGroud[g_iLine + 2][g_iList] = 1;
        }

    }
    else// 长方形是竖着的
    {
        // 背景清零
        g_arrBackGroud[g_iLine - 1][g_iList] = 0;
        g_arrBackGroud[g_iLine + 1][g_iList] = 0;
        g_arrBackGroud[g_iLine + 2][g_iList] = 0;

        if (2 == g_arrBackGroud[g_iLine][g_iList + 1] || 9 == g_iList)
        {
            // 元素复制
            g_arrBackGroud[g_iLine][g_iList - 1] = 1;
            g_arrBackGroud[g_iLine][g_iList - 2] = 1;
            g_arrBackGroud[g_iLine][g_iList - 3] = 1;
            // 中心点标记改变
            g_iList = g_iList - 2;
            // g_iList -= 2;
        }
        else if (2 == g_arrBackGroud[g_iLine][g_iList + 2] || 8 == g_iList)
        {
            // 元素复制
            g_arrBackGroud[g_iLine][g_iList + 1] = 1;
            g_arrBackGroud[g_iLine][g_iList - 1] = 1;
            g_arrBackGroud[g_iLine][g_iList - 2] = 1;
            // 中心点标记改变
            g_iList = g_iList - 1;
            // g_iList -= 1;
        }
        else if (2 == g_arrBackGroud[g_iLine][g_iList - 1] || 0 == g_iList)
        {
            // 元素复制
            g_arrBackGroud[g_iLine][g_iList + 1] = 1;
            g_arrBackGroud[g_iLine][g_iList + 2] = 1;
            g_arrBackGroud[g_iLine][g_iList + 3] = 1;
            // 中心点标记改变
            g_iList = g_iList + 1;
            // g_iList += 1;
        }
        else
        {
            // 元素复制
            g_arrBackGroud[g_iLine][g_iList - 1] = 1;
            g_arrBackGroud[g_iLine][g_iList + 1] = 1;
            g_arrBackGroud[g_iLine][g_iList + 2] = 1;
        }
    }
}

// 长方形方块能否变型条件判断,0不可以变型，1可以变型
int CanLineSqareChangeShape()
{
    int i = 0;
    int j = 0;

    for (i = 1; i < 4; i++)
    {
        if (2 == g_arrBackGroud[g_iLine][g_iList + i] || g_iList + i > 9)
        {
            break;
        }
    }
    for (j = 1; j < 4; j++)
    {
        if (2 == g_arrBackGroud[g_iLine][g_iList - j] || g_iList - j < 0)
        {
            break;
        }
    }

    if ((i-1 + j-1) < 3)
    {
        return 0;
    }

    return 1;
}

// 消除方块
void DestroyOneLineSqare()
{
    int i = 0,
        j = 0;
    int iSum = 0;
    int iTampi = 0;

    for (i = 19; i >= 0; i--)
    {
        for (j = 0; j < 10; j++)
        {
            iSum += g_arrBackGroud[i][j];
        }

        if (20 == iSum)
        {
            // 消除一行
            for (iTampi = i - 1; iTampi >= 0; iTampi--)
            {
                for (j = 0; j < 10; j++)
                {
                    g_arrBackGroud[iTampi + 1][j] = g_arrBackGroud[iTampi][j];
                }

            }
            // 消除一行积分 10
            g_iScore += 10;
            //g_iScore ++;
            // 解决同时多行消除情况
            i = 20;
        }

        iSum = 0;
    }
}

// 显示积分
void ShowScore(HDC hMemDC)
{
    // 字符数组  最高积分
    char strScore[10] = {0};
    char strTopScore[10] = {0};

    Rectangle(hMemDC, 300, 0, 500, 600);

    // 积分显示
    itoa(g_iScore, strScore, 10);
    TextOut(hMemDC, 350, 60, "总积分:", strlen("总积分:"));
    TextOut(hMemDC, 400, 80, strScore, strlen(strScore));

    // 显示提示信息
    itoa(200000, strTopScore, 10);
    TextOut(hMemDC, 350, 130, "最高积分:", strlen("最高积分:"));
    TextOut(hMemDC, 400, 150, strTopScore, strlen(strTopScore));

    TextOut(hMemDC, 350, 200, "开始键:Enter", strlen("开始键:Enter"));
    TextOut(hMemDC, 330, 250, "方向键↑:方块变型", strlen("方向键↑:方块变型"));
    TextOut(hMemDC, 330, 300, "方向键↓:方块加速", strlen("方向键↑:方块变型"));
    TextOut(hMemDC, 330, 350, "方向键←:方块左移", strlen("方向键↑:方块变型"));
    TextOut(hMemDC, 330, 400, "方向键→:方块右移", strlen("方向键↑:方块变型"));

}

// TODO 游戏结束失败
int CanGameOver()
{
    int i = 0;
    for (i = 0; i < 10; i++)
    {
        if (2 == g_arrBackGroud[0][i])
        {
            // 游戏结束
            MessageBox(NULL, "很遗憾，游戏失败了，再接再厉吧!", "提示", MB_OK);//MB_YESNO
            return 0;
        }
        if (200000 == g_iScore)
        {
            // 游戏获胜
            MessageBox(NULL, "恭喜你，通关了，消除了2万行方块!", "提示", MB_OK);//MB_YESNO
            return 0;
        }
    }

    return 1;
}
