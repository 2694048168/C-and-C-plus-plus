#ifndef __C_SOCKET_H__
#define __C_SOCKET_H__

// ====================== 服务器 ======================
int bindSocket(int lfd, unsigned short port);
int setListen(int lfd);
int acceptConn(int lfd, struct sockaddr_in *addr);

// ====================== 客户端 ======================
int connectToHost(int fd, const char *ip, unsigned short port);

// ====================== 共用 ======================
int createSocket();
int sendMsg(int fd, const char *msg);
int recvMsg(int fd, char *msg, int size);
int closeSocket(int fd);

// TCP数据粘包的处理: https://subingwen.cn/linux/tcp-data-package/
int read_n(int fd, char *buf, int size);
int write_n(int fd, const char *msg, int size);

#endif /* __C_SOCKET_H__ */
