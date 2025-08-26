#include "CommunicateModbus.h"

#include <QDebug>

ModbusRTU::ModbusRTU(QObject *parent)
    : ICommunication(parent)
{
}

bool ModbusRTU::connect(const QVariantMap &config)
{
    QMutex Locker locker(&m_ioMutex);
    if (m_device && m_device->state() == QModbusDevice::ConnectedState)
        return true;

    m_device = new QModbusRtuSerialMaster(this);
    const QString portName = config.value("port", "COM1").toString();
    constint baudRate = config.value("baudRate", 9600).toInt();
    const quint8 slaveId = config.value("slaveId", 1).toUInt();

    m_device->setConnectionParameter(QModbusDevice::SerialPortNameParameter, portName);
    m_device->setConnectionParameter(QModbusDevice::SerialBaudRateParameter, baudRate);
    m_device->setTimeout(1000);      // 超时时间1秒
    m_device->setNumberOfRetries(3); // 重试次数

    if (!m_device->connectDevice())
    {
        notifyError(QString("Modbus RTU连接失败: %1").arg(m_device->errorString()));
        return false;
    }

    connect(m_device, &QModbusDevice::stateChanged, [this](QModbusDevice::State state)
            { notifyObservers({{"connected", state == QModbusDevice::ConnectedState}}); });
    connect(m_device, &QModbusDevice::errorOccurred, this, &ModbusRTU::notifyError);

    return true;
}

bool ModbusRTU::readHoldingRegisters(quint8 slaveId, quint16 startAddr, quint16 count)
{
    QMutex Locker locker(&m_ioMutex);
    if (m_device->state() != QModbusDevice::ConnectedState)
        return false;

    QModbusDataUnit request(QModbusDataUnit::HoldingRegisters, startAddr, count);
    QModbusReply *reply = m_device->sendReadRequest(request, slaveId);
    if (!reply)
        return false;

    QEventLoop loop;
    QObject::connect(reply, &QModbusReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if (reply->error() == QModbusDevice::NoError)
    {
        const QVector<quint16> values = reply->result().values();
        QByteArray data(reinterpret_cast<constchar *>(values.data()), values.size() * sizeof(quint16));
        notifyDataReceived(data);
        reply->deleteLater();
        return true;
    }

    notifyError(reply->errorString());
    reply->deleteLater();
    return false;
}
