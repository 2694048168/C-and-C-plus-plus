#include "TCPAssistantClient.h"

#include <QColor.h>

#include <QBrush>
#include <QMetaEnum>
#include <QTextCharFormat>

TCPAssistantClient::TCPAssistantClient(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WidgetTCPAssistantClient())
    , m_pSocket(new QTcpSocket(this))
{
    ui->setupUi(this);

    connect(ui->btn_connect, &QPushButton::clicked, this, &TCPAssistantClient::sl_connectHandler);
    connect(ui->btn_disconnect, &QPushButton::clicked, this, &TCPAssistantClient::sl_disconnectHandler);
    connect(ui->btn_send, &QPushButton::clicked, this, &TCPAssistantClient::sl_clientSendHandler);

    // void QAbstractSocket::connected()
    connect(m_pSocket, &QAbstractSocket::connected, this, &TCPAssistantClient::sl_connectedMsg);
    // void QAbstractSocket::errorOccurred(QAbstractSocket::SocketError socketError)
    connect(m_pSocket, &QAbstractSocket::errorOccurred, this, &TCPAssistantClient::sl_socketErrorMsg);
    connect(m_pSocket, &QIODevice::readyRead, this, &TCPAssistantClient::sl_readDataFromServer);
    // void QAbstractSocket::disconnected()
    connect(m_pSocket, &QAbstractSocket::disconnected, this, &TCPAssistantClient::sl_disconnectedMsg);

    ui->btn_disconnect->setEnabled(false);
    ui->btn_send->setEnabled(false);
}

TCPAssistantClient::~TCPAssistantClient()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

void TCPAssistantClient::sl_connectHandler()
{
    // void	connectToHost(const QHostAddress &address, quint16 port, QIODeviceBase::OpenMode openMode = ReadWrite)
    // auto ip_address = QHostAddress(ui->lineEdit_IP->text());
    auto ip_address = ui->lineEdit_IP->text();
    auto port       = ui->lineEdit_port->text().toUInt();
    // m_pSocket->connectToHost(ip_address, port);
    // m_pSocket->connectToHost(ui->lineEdit_IP->text(), ui->lineEdit_port->text().toInt());

    // void QAbstractSocket::connectToHost(const QString &hostName, quint16 port, QIODeviceBase::OpenMode openMode = ReadWrite, QAbstractSocket::NetworkLayerProtocol protocol = AnyIPProtocol)
    m_pSocket->connectToHost(ip_address, port, QIODeviceBase::ReadWrite, QAbstractSocket::IPv4Protocol);
}

void TCPAssistantClient::sl_connectedMsg()
{
    ui->textEdit_recv->insertPlainText(u8"成功连接服务端\n");
    if (QAbstractSocket::ConnectedState == m_pSocket->state() || QAbstractSocket::ConnectingState == m_pSocket->state())
    {
        ui->lineEdit_IP->setEnabled(false);
        ui->lineEdit_port->setEnabled(false);
        ui->btn_connect->setEnabled(false);
        ui->btn_disconnect->setEnabled(true);

        if (!ui->btn_send->isEnabled())
            ui->btn_send->setEnabled(true);
    }
}

void TCPAssistantClient::sl_socketErrorMsg(QAbstractSocket::SocketError socketError)
{
    // qDebug() << "[Error] ---> " << socketError;
    // QAbstractSocket::UnsupportedSocketOperationError
    // TODO: 本地设置代理或者没有SSL加密,导致连接失败

    // ui->textEdit_recv->insertPlainText(u8"[Error] ---> 连接服务端失败\n");

    auto cursor_text = ui->textEdit_recv->textCursor();

    QTextCharFormat format;
    format.setForeground(QBrush(QColor(Qt::red)));
    cursor_text.setCharFormat(format);

    QMetaEnum metaEnum  = QMetaEnum::fromType<QAbstractSocket::SocketError>();
    QString   str       = metaEnum.valueToKey(socketError);
    QString   error_str = u8"[Socket Error] ---> " + str;
    cursor_text.insertText(error_str + "\n");
}

void TCPAssistantClient::sl_readDataFromServer()
{
    auto recvData = m_pSocket->readAll();

    // ui->textEdit_recv->insertPlainText(recvData + "\n");
    auto cursor_text = ui->textEdit_recv->textCursor();

    QTextCharFormat format;
    format.setForeground(QBrush(QColor(Qt::green)));
    cursor_text.setCharFormat(format);
    cursor_text.insertText(recvData + "\n");
}

void TCPAssistantClient::sl_clientSendHandler()
{
    QByteArray sendData = ui->textEdit_send->toPlainText().toUtf8();
    m_pSocket->write(sendData);

    auto cursor_text = ui->textEdit_recv->textCursor();

    QTextCharFormat format;
    format.setForeground(QBrush(QColor(Qt::yellow)));
    cursor_text.setCharFormat(format);
    cursor_text.insertText(sendData + "\n");
}

void TCPAssistantClient::sl_disconnectHandler()
{
    // void QAbstractSocket::disconnectFromHost()
    m_pSocket->disconnectFromHost();
}

void TCPAssistantClient::sl_disconnectedMsg()
{
    ui->textEdit_recv->insertPlainText(u8"成功断开服务端\n");
    ui->lineEdit_IP->setEnabled(true);
    ui->lineEdit_port->setEnabled(true);
    ui->btn_connect->setEnabled(true);
    ui->btn_disconnect->setEnabled(false);
}
