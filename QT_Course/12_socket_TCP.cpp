/**
 * @file 12_socket_TCP.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在标准C++没有提供专门用于套接字通信的类,所以只能使用操作系统提供的基于C的API函数,
 * 基于这些C的API函数也可以封装自己的C++类 C++套接字类的封装.
 * *QT是C++的一个框架并且里边提供了用于套接字通信的类(TCP、UDP),使用相关类的进行TCP通信.
 * 使用Qt提供的类进行基于TCP的套接字通信需要用到两个类:
 * ?1. QTcpServer: 服务器类,用于监听客户端连接以及和客户端建立连接;
 * ?2. QTcpSocket: 通信的套接字类, 客户端、服务器端都需要使用;
 * ?这两个套接字通信类都属于网络模块network;
 * 
https://subingwen.cn/qt/socket-tcp/#1-1-%E5%85%AC%E5%85%B1%E6%88%90%E5%91%98%E5%87%BD%E6%95%B0
 * 
 * !服务器端----通信流程
1. 创建套接字服务器QTcpServer对象;
2. 通过QTcpServer对象设置监听，即：QTcpServer::listen();
3. 基于QTcpServer::newConnection()信号检测是否有新的客户端连接;
4. 如果有新的客户端连接调用QTcpSocket *QTcpServer::nextPendingConnection()得到通信的套接字对象;
5. 使用通信的套接字对象QTcpSocket和客户端进行通信;
// ========================================
 * 
 * !客户端----通信流程
1. 创建通信的套接字类QTcpSocket对象;
2. 使用服务器端绑定的IP和端口连接服务器QAbstractSocket::connectToHost();
3. 使用QTcpSocket对象和服务器进行通信;
// ========================================
 * 
 */

#include "src/TCP_client.h"
#include "src/TCP_server.h"

#include <QApplication>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    //*==============================================
    TCPServer tcp_server;
    tcp_server.show();

    TCPClient tcp_client;
    tcp_client.show();

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
