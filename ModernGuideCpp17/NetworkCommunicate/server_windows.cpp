/**
 * @file server_windows.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Windows OS 网络通信服务端
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// 使用包含的头文件
#include <winsock2.h>

// 使用的套接字库
// ws2_32.dll

// ===================================
int main(int argc, const char **argv)
{
    // 0. 初始化Winsock库
    WSAData wsa;
    // 初始化套接字库
    WSAStartup(MAKEWORD(2, 2), &wsa);

    // 1. 创建监听的套接字
    SOCKET lfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (lfd == INVALID_SOCKET)
    {
        std::cout << "create listen socket NOT successfully\n";
        return -1;
    }

    // 2. 将socket()返回值和本地的IP端口绑定到一起
    struct sockaddr_in saddr;
    saddr.sin_family      = AF_INET;
    saddr.sin_addr.s_addr = INADDR_ANY;
    saddr.sin_port        = 10000;
    int len               = sizeof(saddr);

    int ret = bind(lfd, (const struct sockaddr *)&saddr, len);
    if (ret == SOCKET_ERROR)
    {
        std::cout << "bind NOT successfully\n";
        return -1;
    }

    // 3. 设置监听
    ret = listen(lfd, 128);
    if (ret == SOCKET_ERROR)
    {
        std::cout << "listen NOT successfully\n";
        return -1;
    }

    // 4. 阻塞等待并接受客户端连接
    struct sockaddr_in caddr;

    int clen = sizeof(caddr);

    SOCKET cfd = accept(lfd, (struct sockaddr *)&caddr, &clen);
    if (cfd == INVALID_SOCKET)
    {
        std::cout << "create accept(communication) socket NOT successfully\n";
        return -1;
    }

    // 打印客户端的地址信息
    std::cout << "客户端的IP地址: " << ntohs(caddr.sin_addr.s_addr) << "端口: " << ntohs(caddr.sin_port) << '\n';

    // 5. 和客户端通信
    while (true)
    {
        // 接收数据
        char buf[1024];
        memset(buf, 0, sizeof(buf));
        int len = recv(cfd, buf, sizeof(buf), 0);
        if (len > 0)
        {
            std::cout << "客户端say: %s\n" << buf;
            send(cfd, buf, len, 0);
        }
        else if (len == 0)
        {
            std::cout << "客户端断开了连接...\n";
            break;
        }
        else
        {
            std::cout << "read SOCKET_ERROR\n";
            break;
        }
    }

    ret = closesocket(lfd);
    if (ret == SOCKET_ERROR)
    {
        std::cout << "close listen socket NOT successfully\n";
        return -1;
    }
    ret = closesocket(cfd);
    if (ret == SOCKET_ERROR)
    {
        std::cout << "close communication socket NOT successfully\n";
        return -1;
    }

    // 注销Winsock相关库
    WSACleanup();

    return 0;
}
