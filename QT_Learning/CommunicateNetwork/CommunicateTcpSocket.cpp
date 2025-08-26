#include "CommunicateTcpSocket.h"

#include <QDebug>

NetworkSocket::NetworkSocket(QObject *parent)
    : ICommunication(parent)
{
}

bool NetworkSocket::connect(const QVariantMap &config)
{
    QMutex Locker locker(&m_ioMutex);
    if (m_socket && m_socket->state() == QTcpSocket::ConnectedState)
        return true;

    m_socket = newQTcpSocket(this);
    const QString host = config.value("host", "192.168.1.200").toString();
    const quint16 port = config.value("port", 8080).toUInt();

    connect(m_socket, &QTcpSocket::connected, [this]()
            { notifyObservers({{"connected", true}, {"host", m_socket->peerName()}}); });
    connect(m_socket, &QTcpSocket::disconnected, this, &NetworkSocket::onDisconnected);
    connect(m_socket, &QTcpSocket::readyRead, this, &NetworkSocket::onReadyRead);
    connect(m_socket, QOverload<QAbstractSocket::SocketError>::of(&QTcpSocket::errorOccurred),
            this, &NetworkSocket::notifyError);

    m_socket->connectToHost(host, port);
    return true;
}

void NetworkSocket::onReadyRead()
{
    QByteArray data = m_socket->readAll();
    notifyDataReceived(data); // 通知观察者数据接收
}
