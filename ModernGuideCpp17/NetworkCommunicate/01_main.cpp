/**
 * @file 01_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 基于C++的封装
 * 编写C++程序应当遵循面向对象三要素: 封装、继承、多态.
 * 封装之后的类可以隐藏掉某些属性使操作更简单并且类的功能要单一,
 * 如果要代码重用可以进行类之间的继承,如果要让函数的使用更加灵活可以使用多态.
 * ?因此需要封装两个类: 客户端类和服务器端的类.
 * 
 * *基于TCP的套接字通信分为两部分: 服务器端通信和客户端通信.
 * 只要掌握了通信流程,封装出对应的功能函数也就不在话下了,通信流程:
 * ?====服务器端
 * 1. 创建用于监听的套接字;
 * 2. 将用于监听的套接字和本地的IP以及端口进行绑定;
 * 3. 启动监听;
 * 4. 等待并接受新的客户端连接,连接建立得到用于通信的套接字和客户端的IP,端口信息;
 * 5. 使用得到的通信的套接字和客户端通信(接收和发送数据);
 * 6. 通信结束,关闭套接字(监听 + 通信);
 * ?====客户端
 * 1. 创建用于通信的套接字;
 * 2. 使用服务器端绑定的IP和端口连接服务器;
 * 3. 使用通信的套接字和服务器通信(发送和接收数据);
 * 4. 通信结束,关闭套接字(通信);
 * 
 * ? https://subingwen.cn/linux/socket/
 * ? https://subingwen.cn/linux/three-four/
 * ? https://subingwen.cn/linux/tcp-status/
 * ? https://subingwen.cn/linux/tcp-data-package/
 * ? https://subingwen.cn/linux/socket-class/
 * 
 */

#include "01_cpp_socket.h"

#include <iostream>

// =============== 服务端 ===============
struct SockInfo
{
    TcpServer         *s;
    TcpSocket         *tcp;
    struct sockaddr_in addr;
};

void *working(void *arg)
{
    struct SockInfo *pinfo = static_cast<struct SockInfo *>(arg);
    // 连接建立成功, 打印客户端的IP和端口信息
    char             ip[32];
    printf("客户端的IP: %s, 端口: %d\n", inet_ntop(AF_INET, &pinfo->addr.sin_addr.s_addr, ip, sizeof(ip)),
           ntohs(pinfo->addr.sin_port));

    // 5. 通信
    while (1)
    {
        printf("接收数据: .....\n");
        std::string msg = pinfo->tcp->recvMsg();
        if (!msg.empty())
        {
            std::cout << msg << std::endl << std::endl << std::endl;
        }
        else
        {
            break;
        }
    }
    delete pinfo->tcp;
    delete pinfo;
    return nullptr;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    // =============== 客户端 ===============
    // 1. 创建通信的套接字
    TcpSocket tcp;

    // 2. 连接服务器IP port
    int ret = tcp.connectToHost("192.168.237.131", 1000);
    if (ret == -1)
    {
        return -1;
    }

    // 3. 通信
    int  fd1    = open("english.txt", O_RDONLY);
    int  length = 0;
    char tmp[100];
    memset(tmp, 0, sizeof(tmp));
    while ((length = read(fd1, tmp, sizeof(tmp))) > 0)
    {
        // 发送数据
        tcp.sendMsg(std::string(tmp, length));

        std::cout << "send Msg: " << std::endl;
        std::cout << tmp << std::endl << std::endl << std::endl;
        memset(tmp, 0, sizeof(tmp));

        // 接收数据
        usleep(300);
    }

    sleep(10);

    // =============== 服务端 ===============
    // 1. 创建监听的套接字
    TcpServer s;
    // 2. 绑定本地的IP port并设置监听
    s.setListen(10000);
    // 3. 阻塞并等待客户端的连接
    while (1)
    {
        SockInfo  *info = new SockInfo;
        TcpSocket *tcp  = s.acceptConn(&info->addr);
        if (tcp == nullptr)
        {
            std::cout << "重试...." << std::endl;
            continue;
        }
        // 创建子线程
        pthread_t tid;
        info->s   = &s;
        info->tcp = tcp;

        pthread_create(&tid, NULL, working, info);
        pthread_detach(tid);
    }

    return 0;
}
