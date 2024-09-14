#include "TCP_server.h"

#include "ui_TCP_server.h"

TCPServer::TCPServer(QWidget *parent)
    : ui(new Ui::TCPServer)
{
    ui->setupUi(this);
    // 创建 QTcpServer 对象
    m_pServer = new QTcpServer(this);
    // 检测是否有新的客户端连接
    connect(m_pServer, &QTcpServer::newConnection, this,
            [=]()
            {
                m_pTcp = m_pServer->nextPendingConnection();
                ui->textEdit_communicate->append("成功和客户端建立了新的连接...");

                ui->label_statusConnect->setStyleSheet(m_green_SheetStyle);

                // 检测是否有客户端数据
                connect(m_pTcp, &QTcpSocket::readyRead, this,
                        [=]()
                        {
                            // 接收数据
                            QString recvMsg = m_pTcp->readAll();
                            ui->textEdit_communicate->append("客户端Say: " + recvMsg);
                        });
                // 客户端断开了连接
                connect(m_pTcp, &QTcpSocket::disconnected, this,
                        [=]()
                        {
                            ui->textEdit_communicate->append("客户端已经断开了连接...");
                            m_pTcp->deleteLater();

                            ui->label_statusConnect->setStyleSheet(m_red_SheetStyle);
                        });
            });
}

TCPServer::~TCPServer()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

// 启动服务器端的服务按钮
// void TCPServer::on_startServer_clicked()
void TCPServer::sl_startServer_clicked()
{
    unsigned short port = ui->lineEdit_port->text().toInt();
    // 设置服务器监听
    m_pServer->listen(QHostAddress::Any, port);
    ui->btn_startServer->setEnabled(false);
}

// 点击发送数据按钮
// void TCPServer::on_sendMsg_clicked()
void TCPServer::sl_sendMsg_clicked()
{
    QString sendMsg = ui->textEdit_sendData->toPlainText();
    m_pTcp->write(sendMsg.toUtf8());
    ui->textEdit_communicate->append("服务器Say: " + sendMsg);
    ui->textEdit_sendData->clear();
}
