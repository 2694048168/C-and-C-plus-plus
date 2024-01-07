/**
 * @file ThreadBase.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Windows Thread API into a class
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __THREAD_BASE_HPP__
#define __THREAD_BASE_HPP__

#include <process.h>
#include <windows.h>

class ThreadBase
{
public:
    ThreadBase();
    ~ThreadBase();

    //开始线程
    bool Start();

    //阻塞等待子线程结束
    void Join(int timeout = -1);

    //运行函数
    virtual void Run();

    //设置线程退出
    void SetThreadEnd();

    //设置线程CPU亲和力
    void SetAffinityMask(DWORD_PTR mask);

    //设置线程优先级 相对优先级
    void SetPriority(int nPriority);

    //激活线程
    void ResumeThread();

    //挂起线程
    void SuspendThread();

private:
    static unsigned int WINAPI StaticThreadFunc(void *arg);

private:
    unsigned int m_ThreadID; //线程ID
    char        *m_Name;     //线程名称
    HANDLE       m_Handle;   //线程句柄

    bool m_Run;   //线程是否正在运行
    bool m_isEnd; //是否结束线程

    HANDLE g_hThreadEvent; //事件
};

#endif // !__THREAD_BASE_HPP__