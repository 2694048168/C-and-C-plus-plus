#include "ThreadBase.hpp"

ThreadBase::ThreadBase()
{
    //线程句柄
    m_Handle = (HANDLE)_beginthreadex(NULL, 0, StaticThreadFunc, this, 0, &m_ThreadID);

    m_ThreadID = -1;   //线程ID
    m_Name     = NULL; //线程名称

    m_Run = false; //线程是否运行

    //初始化事件
    g_hThreadEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
}

ThreadBase::~ThreadBase() {}

unsigned int ThreadBase::StaticThreadFunc(void *arg)
{
    ThreadBase *thread = (ThreadBase *)arg;

    while (true)
    {
        thread->m_Run = false;
        WaitForSingleObject(thread->g_hThreadEvent, INFINITE); //等待事件被触发
        thread->m_Run = true;

        thread->Run(); //线程处理函数

        if (thread->m_isEnd) //是否退出线程
        {
            break;
        }
    }

    return 0;
}

//开始线程
bool ThreadBase::Start()
{
    //如果线程正在运行则返回
    if (m_Run || NULL == m_Handle)
    {
        return false;
    }

    m_Run = true;             //重置标志位
    SetEvent(g_hThreadEvent); //触发事件

    return m_Run;
}

//阻塞等待子线程结束
void ThreadBase::Join(int timeout)
{
    if (NULL == m_Handle || !m_Run)
    {
        return;
    }

    if (timeout <= 0)
    {
        timeout = INFINITE;
    }

    //阻塞等待线程结束
    ::WaitForSingleObject(m_Handle, timeout);
}

//设置线程退出
void ThreadBase::SetThreadEnd()
{
    m_isEnd = true;
}

//设置线程CPU亲和力
/**
 * @brief 
 * // 获取QT线程id
 * quint64 threadId = (quint64)QThread::currentThreadId();
 * // 线程id=>线程句柄
 * HANDLE handle = OpenThread(THREAD_ALL_ACCESS, false, threadId);
 * // 设置线程亲和性
 * SetThreadAffinityMask(handle, 0x02);
 */
void ThreadBase::SetAffinityMask(DWORD_PTR mask)
{
    ::SetThreadAffinityMask(m_Handle, mask);
}

//激活线程
void ThreadBase::ResumeThread()
{
    if (NULL == m_Handle || !m_Run)
    {
        return;
    }
    ::ResumeThread(m_Handle);
}

//挂起线程
void ThreadBase::SuspendThread()
{
    if (NULL == m_Handle || m_Run)
    {
        return;
    }
    ::SuspendThread(m_Handle);
}

//运行函数
void ThreadBase::Run() {}