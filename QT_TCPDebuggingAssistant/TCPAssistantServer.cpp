#include "TCPAssistantServer.h"

#include "ComboBoxRefresh.h"

#include <QList>
#include <QMessageBox>
#include <QNetworkInterface>
#include <QTcpSocket>

TCPAssistantServer::TCPAssistantServer(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WidgetTCPAssistantServer())
    , m_pServer{new QTcpServer(this)}
    , m_currentClientIndex{0}
{
    ui->setupUi(this);

    // QList<QHostAddress> QNetworkInterface::allAddresses()
    QList<QHostAddress> address_list = QNetworkInterface::allAddresses();
    for (const auto &ip_addr : address_list)
    {
        // ui->comboBox_IP->addItem(ip_addr.toString());
        // enum QAbstractSocket::NetworkLayerProtocol
        if (QAbstractSocket::IPv4Protocol == ip_addr.protocol())
            ui->comboBox_IP->addItem(ip_addr.toString());
    }

    // void QTcpServer::newConnection()
    connect(m_pServer, &QTcpServer::newConnection, this, &TCPAssistantServer::sl_newClientConnect);

    connect(ui->btn_listen, &QPushButton::clicked, this, &TCPAssistantServer::sl_listenHandler);
    connect(ui->btn_stopListen, &QPushButton::clicked, this, &TCPAssistantServer::sl_stopListenHandler);
    connect(ui->btn_disconnect, &QPushButton::clicked, this, &TCPAssistantServer::sl_disconnectHandler);
    connect(ui->btn_send, &QPushButton::clicked, this, &TCPAssistantServer::sl_serverSendHandler);

    ui->btn_stopListen->setEnabled(false);
    ui->btn_disconnect->setEnabled(false);
    ui->btn_send->setEnabled(false);

    connect(ui->comboBox_clients, &ComboBoxRefresh::comboBox_clicked, this,
            &TCPAssistantServer::sl_clientsRefreshHandler);
}

TCPAssistantServer::~TCPAssistantServer()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

void TCPAssistantServer::sl_newClientConnect()
{
    // bool QTcpServer::hasPendingConnections() const
    if (m_pServer->hasPendingConnections())
    {
        // QTcpSocket *QTcpServer::nextPendingConnection()
        QTcpSocket *pSocket = m_pServer->nextPendingConnection();

        auto IP_str   = pSocket->peerAddress().toString();
        auto port_str = u8" 客户端端口: " + QString::number(pSocket->peerPort());
        auto msg_str  = u8"客户端IP地址: " + IP_str + port_str + "\n";
        ui->textEdit_recv->insertPlainText(msg_str);

        // void QIODevice::readyRead()
        connect(pSocket, &QIODevice::readyRead, this, &TCPAssistantServer::sl_socketReadyReadHandler);
        // void QAbstractSocket::disconnected()
        connect(pSocket, &QAbstractSocket::disconnected, this, &TCPAssistantServer::sl_clientDisconnectedHandler);
        // void QAbstractSocket::stateChanged(QAbstractSocket::SocketState socketState)

        if (!ui->btn_send->isEnabled())
            ui->btn_send->setEnabled(true);
    }
}

void TCPAssistantServer::sl_listenHandler()
{
    // bool QTcpServer::listen(const QHostAddress &address = QHostAddress::Any, quint16 port = 0)
    // if (!m_pServer->listen(QHostAddress("127.0.0.1"), 9999))
    int  port = ui->lineEdit_port->text().toInt();
    auto ip   = QHostAddress(ui->comboBox_IP->currentText());
    // if (!m_pServer->listen(QHostAddress::Any, 9999))
    // if (!m_pServer->listen(QHostAddress::Any, port))
    if (!m_pServer->listen(ip, port))
    {
        ui->textEdit_recv->insertPlainText(tr(u8"监听失败\n"));

        QMessageBox msg_box;
        msg_box.setWindowTitle(tr(u8"提示信息"));
        msg_box.setIcon(QMessageBox::Critical);
        msg_box.setText(tr(u8"通信端口被占用, 请检查确认? "));
        msg_box.exec();

        return;
    }

    ui->btn_listen->setEnabled(false);
    ui->btn_stopListen->setEnabled(true);
    ui->btn_disconnect->setEnabled(true);
}

void TCPAssistantServer::sl_stopListenHandler()
{
    // 释放资源
    QList<QTcpSocket *> tcpSocketClients = m_pServer->findChildren<QTcpSocket *>();
    for (const auto &pSocket : tcpSocketClients)
    {
        pSocket->close();
    }

    m_pServer->close();

    ui->btn_listen->setEnabled(true);
    ui->btn_stopListen->setEnabled(false);
    ui->btn_disconnect->setEnabled(false);
}

void TCPAssistantServer::sl_disconnectHandler()
{
    sl_disconnectHandler();
    if (m_pServer)
    {
        delete m_pServer;
        m_pServer = nullptr;
    }
    this->close();
}

void TCPAssistantServer::sl_socketReadyReadHandler()
{
    QTcpSocket *pSocket  = qobject_cast<QTcpSocket *>(sender());
    QByteArray  recvData = pSocket->readAll();
    ui->textEdit_recv->insertPlainText(u8"客户端: " + recvData + "\n");
}

void TCPAssistantServer::sl_clientDisconnectedHandler()
{
    QTcpSocket *pSocket = qobject_cast<QTcpSocket *>(sender());
    ui->textEdit_recv->insertPlainText(u8"客户端断开连接\n");
    pSocket->deleteLater(); // 资源回收
}

void TCPAssistantServer::sl_serverSendHandler()
{
    // QList<T>	findChildren(const QString &name, Qt::FindChildOptions options = Qt::FindChildrenRecursively) const
    QList<QTcpSocket *> tcpSocketClients = m_pServer->findChildren<QTcpSocket *>();

    if ("All" == ui->comboBox_clients->currentText())
    {
        for (const auto &pSocket : tcpSocketClients)
        {
            pSocket->write(ui->textEdit_send->toPlainText().toStdString().c_str());
        }
    }
    else
    {
        tcpSocketClients[m_currentClientIndex]->write(ui->textEdit_send->toPlainText().toStdString().c_str());
    }
}

void TCPAssistantServer::sl_clientsRefreshHandler()
{
    ui->comboBox_clients->clear();
    // QList<T>	findChildren(const QString &name, Qt::FindChildOptions options = Qt::FindChildrenRecursively) const
    QList<QTcpSocket *> tcpSocketClients = m_pServer->findChildren<QTcpSocket *>();
    for (const auto &pSocket : tcpSocketClients)
    {
        // if (pSocket)
        // if (pSocket->peerPort() != 0)
        ui->comboBox_clients->addItem(QString::number(pSocket->peerPort()));
    }
    ui->comboBox_clients->addItem("All");
}

void TCPAssistantServer::on_comboBox_clients_activated(int index)
{
    m_currentClientIndex = index;
}