#include "01_cpp_socket.h"

TcpSocket::TcpSocket()
{
    m_fd = socket(AF_INET, SOCK_STREAM, 0);
}

TcpSocket::TcpSocket(int socket)
{
    m_fd = socket;
}

TcpSocket::~TcpSocket()
{
    if (m_fd > 0)
    {
        close(m_fd);
    }
}

int TcpSocket::connectToHost(std::string ip, unsigned short port)
{
    // 连接服务器IP port
    struct sockaddr_in saddr;
    saddr.sin_family = AF_INET;
    saddr.sin_port   = htons(port);
    inet_pton(AF_INET, ip.data(), &saddr.sin_addr.s_addr);
    int ret = connect(m_fd, (struct sockaddr *)&saddr, sizeof(saddr));
    if (ret == -1)
    {
        perror("connect");
        return -1;
    }
    std::cout << "成功和服务器建立连接..." << std::endl;
    return ret;
}

int TcpSocket::sendMsg(std::string msg)
{
    // 申请内存空间: 数据长度 + 包头4字节(存储数据长度)
    char *data   = new char[msg.size() + 4];
    int   bigLen = htonl(msg.size());
    std::memcpy(data, &bigLen, 4);
    std::memcpy(data + 4, msg.data(), msg.size());
    // 发送数据
    int ret = writen(data, msg.size() + 4);
    delete[] data;
    return ret;
}

std::string TcpSocket::recvMsg()
{
    // 接收数据
    // 1. 读数据头
    int len = 0;
    readn((char *)&len, 4);
    len = ntohl(len);
    std::cout << "数据块大小: " << len << std::endl;

    // 根据读出的长度分配内存
    char *buf = new char[len + 1];
    int   ret = readn(buf, len);
    if (ret != len)
    {
        return std::string();
    }
    buf[len] = '\0';
    string retStr(buf);
    delete[] buf;

    return retStr;
}

int TcpSocket::readn(char *buf, int size)
{
    int   nread = 0;
    int   left  = size;
    char *p     = buf;

    while (left > 0)
    {
        if ((nread = read(m_fd, p, left)) > 0)
        {
            p += nread;
            left -= nread;
        }
        else if (nread == -1)
        {
            return -1;
        }
    }
    return size;
}

int TcpSocket::writen(const char *msg, int size)
{
    int         left   = size;
    int         nwrite = 0;
    const char *p      = msg;

    while (left > 0)
    {
        if ((nwrite = write(m_fd, msg, left)) > 0)
        {
            p += nwrite;
            left -= nwrite;
        }
        else if (nwrite == -1)
        {
            return -1;
        }
    }
    return size;
}

TcpServer::TcpServer()
{
    m_fd = socket(AF_INET, SOCK_STREAM, 0);
}

TcpServer::~TcpServer()
{
    close(m_fd);
}

int TcpServer::setListen(unsigned short port)
{
    struct sockaddr_in saddr;
    saddr.sin_family      = AF_INET;
    saddr.sin_port        = htons(port);
    saddr.sin_addr.s_addr = INADDR_ANY; // 0 = 0.0.0.0
    int ret               = bind(m_fd, (struct sockaddr *)&saddr, sizeof(saddr));
    if (ret == -1)
    {
        perror("bind");
        return -1;
    }
    cout << "套接字绑定成功, ip: " << inet_ntoa(saddr.sin_addr) << ", port: " << port << endl;

    ret = listen(m_fd, 128);
    if (ret == -1)
    {
        perror("listen");
        return -1;
    }
    cout << "设置监听成功..." << endl;

    return ret;
}

TcpSocket *TcpServer::acceptConn(sockaddr_in *addr)
{
    if (addr == NULL)
    {
        return nullptr;
    }

    socklen_t addrlen = sizeof(struct sockaddr_in);
    int       cfd     = accept(m_fd, (struct sockaddr *)addr, &addrlen);
    if (cfd == -1)
    {
        perror("accept");
        return nullptr;
    }
    printf("成功和客户端建立连接...\n");
    return new TcpSocket(cfd);
}
