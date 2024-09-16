#ifndef __CPP_SOCKET_H__
#define __CPP_SOCKET_H__

#include <string>

class TcpClient
{
public:
    TcpClient();
    ~TcpClient();
    int connectToHost(std::string ip, unsigned short port);

    int    sendMsg(std::string msg);
    string recvMsg();

private:
    int readn(char *buf, int size);
    int writen(const char *msg, int size);

private:
    int cfd; // 通信的套接字
};

class TcpServer_
{
public:
    TcpServer_();
    ~TcpServer_();

    int setListen(unsigned short port);
    int acceptConn(struct sockaddr_in *addr);

    int    sendMsg(std::string msg);
    string recvMsg();

private:
    int readn(char *buf, int size);
    int writen(const char *msg, int size);

private:
    int lfd; // 监听的套接字
    int cfd; // 通信的套接字
};

// =====通信类
// 套接字通信类既可以在客户端使用，也可以在服务器端使用，职责是接收和发送数据包
class TcpSocket
{
public:
    TcpSocket();
    TcpSocket(int socket);
    ~TcpSocket();
    int    connectToHost(std::string ip, unsigned short port);
    int    sendMsg(std::string msg);
    string recvMsg();

private:
    int readn(char *buf, int size);
    int writen(const char *msg, int size);

private:
    int m_fd; // 通信的套接字
};

// ====服务器类
// 服务器类主要用于套接字通信的服务器端，并且没有通信能力，当服务器和客户端的新连接建立之后，
// 需要通过TcpSocket类的带参构造将通信的描述符包装成一个通信对象，这样就可以使用这个对象和客户端通信了。
class TcpServer
{
public:
    TcpServer();
    ~TcpServer();
    int        setListen(unsigned short port);
    TcpSocket *acceptConn(struct sockaddr_in *addr = nullptr);

private:
    int m_fd; // 监听的套接字
};

#endif /* __CPP_SOCKET_H__ */