#include "TCP_client.h"

#include "ui_TCP_client.h"

TCPClient::TCPClient(QWidget *parent)
    : ui(new Ui::TCPClient)
{
    ui->setupUi(this);

    // 创建通信的套接字对象
    m_pTcp = new QTcpSocket(this);
    // 检测服务器是否回复了数据
    connect(m_pTcp, &QTcpSocket::readyRead,
            [=]()
            {
                // 接收服务器发送的数据
                QByteArray recvMsg = m_pTcp->readAll();
                ui->textEdit_communicate->append("服务器Say: " + recvMsg);
            });

    // 检测是否和服务器是否连接成功了
    connect(m_pTcp, &QTcpSocket::connected, this,
            [=]()
            {
                ui->textEdit_communicate->append("恭喜, 连接服务器成功!!!");
                ui->label_statusConnect->setStyleSheet(m_green_SheetStyle);
            });

    // 检测服务器是否和客户端断开了连接
    connect(m_pTcp, &QTcpSocket::disconnected, this,
            [=]()
            {
                ui->textEdit_communicate->append("服务器已经断开了连接, ...");
                ui->btn_connectServer->setEnabled(true);
                ui->btn_disconnectServer->setEnabled(false);

                ui->label_statusConnect->setStyleSheet(m_red_SheetStyle);
            });
}

TCPClient::~TCPClient()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

// 连接服务器按钮按下之后的处理动作
// void TCPClient::on_connectServer_clicked()
void TCPClient::sl_connectServer_clicked()
{
    QString        ip   = ui->lineEdit_IP->text();
    unsigned short port = ui->lineEdit_port->text().toInt();
    // 连接服务器
    m_pTcp->connectToHost(QHostAddress(ip), port);
    ui->btn_connectServer->setEnabled(false);
    ui->btn_disconnectServer->setEnabled(true);
}

// 发送数据按钮按下之后的处理动作
// void TCPClient::on_sendMsg_clicked()
void TCPClient::sl_sendMsg_clicked()
{
    QString sendMsg = ui->textEdit_sendData->toPlainText();
    m_pTcp->write(sendMsg.toUtf8());
    ui->textEdit_communicate->append("客户端Say: " + sendMsg);
    ui->textEdit_sendData->clear();
}

// 断开连接按钮被按下之后的处理动作
// void TCPClient::on_disconnect_clicked()
void TCPClient::sl_disconnect_clicked()
{
    m_pTcp->close();
    ui->btn_connectServer->setEnabled(true);
    ui->btn_disconnectServer->setEnabled(false);
}
