#include "CommunicateSerialPort.h"

#include <QDebug>

SerialPort::SerialPort(QObject *parent)
    : ICommunication(parent)
{
}

SerialPort::~SerialPort()
{
    disconnect();
}

bool SerialPort::connect(const QVariantMap &config)
{
    QMutex Locker locker(&m_ioMutex);
    if (m_serial && m_serial->isOpen())
        return true;

    m_serial = new QSerialPort(this);
    // 从配置中读取参数（支持动态配置）
    const QString portName = config.value("port", "").toString();
    const int baudRate = config.value("baudRate", 115200).toInt();
    const QSerialPort::DataBits dataBits = static_cast<QSerialPort::DataBits>(
        config.value("dataBits", QSerialPort::Data8).toInt());
    const QSerialPort::Parity parity = static_cast<QSerialPort::Parity>(
        config.value("parity", QSerialPort::NoParity).toInt());
    const QSerialPort::StopBits stopBits = static_cast<QSerialPort::StopBits>(
        config.value("stopBits", QSerialPort::OneStop).toInt());

    m_serial->setPortName(portName);
    m_serial->setBaudRate(baudRate);
    m_serial->setDataBits(dataBits);
    m_serial->setParity(parity);
    m_serial->setStopBits(stopBits);
    m_serial->setFlowControl(QSerialPort::NoFlowControl);

    if (!m_serial->open(QIODevice::ReadWrite))
    {
        notifyError(QString("串口打开失败：%1").arg(m_serial->errorString()));
        return false;
    }

    connect(m_serial, &QSerialPort::readyRead, this, &SerialPort::onReadyRead);
    notifyObservers({{"connected", true}, {"port", portName}});
    return true;
}

void SerialPort::disconnect()
{
    QMutex Locker locker(&m_ioMutex);
    if (m_serial && m_serial->isOpen())
    {
        m_serial->close();
        delete m_serial;
        m_serial = nullptr;
        notifyObservers({{"connected", false}});
    }
}

qint64 SerialPort::write(const QByteArray &data)
{
    QMutex Locker locker(&m_ioMutex);
    if (!m_serial || !m_serial->isOpen())
    {
        notifyError("串口未连接");
        return -1;
    }
    return m_serial->write(data);
}

QByteArray SerialPort::read(qint32 timeout)
{
    QMutex Locker locker(&m_ioMutex);
    if (!m_serial || !m_serial->isOpen())
        returnQByteArray();

    if (m_serial->waitForReadyRead(timeout))
    {
        return m_serial->readAll();
    }
    return QByteArray();
}

QVariantMap SerialPort::status() const
{
    QMutex Locker locker(&m_ioMutex);
    return {
        {"connected", m_serial && m_serial->isOpen()},
        {"port", m_serial ? m_serial->portName() : ""},
        {"bytesAvailable", m_serial ? m_serial->bytesAvailable() : 0}};
}

void SerialPort::onReadyRead()
{
    QByteArray data = m_serial->readAll();
    notifyDataReceived(data); // 通知观察者数据接收
}
