#include "CommunicateCanBus.h"

#include <QDebug>

CanBus::CanBus(QObject *parent)
    : ICommunication(parent)
{
}

bool CanBus::connect(const QVariantMap &config)
{
    QMutex Locker locker(&m_ioMutex);
    if (m_device && m_device->state() == QCanBusDevice::ConnectedState)
        return true;

    m_device = QCanBusDevice::instance(config.value("interface", "can0").toString());
    if (!m_device)
    {
        notifyError("未找到CAN接口");
        return false;
    }

    m_device->setConfigurationParameter(QCanBusDevice::BitRateKey, config.value("bitRate", 500000).toInt());
    m_device->setConfigurationParameter(QCanBusDevice::LoopbackKey, 0);

    if (!m_device->connectDevice())
    {
        notifyError(QString("CAN连接失败：%1").arg(m_device->errorString()));
        return false;
    }

    connect(m_device, &QCanBusDevice::framesReceived, this, &CanBus::onFramesReceived);
    notifyObservers({{"connected", true}, {"interface", config.value("interface").toString()}});
    return true;
}

void CanBus::onFramesReceived()
{
    const QVector<QCanBusFrame> frames = m_device->readAllFrames();
    for (const QCanBusFrame &frame : frames)
    {
        if (frame.payloadLength() > 0)
        {
            QByteArray data(frame.data(), frame.payloadLength());
            notifyDataReceived(data); // 通知观察者数据接收
        }
    }
}
