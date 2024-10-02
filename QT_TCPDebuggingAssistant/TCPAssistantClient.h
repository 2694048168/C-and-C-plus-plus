/**
 * @file TCPAssistantClient.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "ui_TCPAssistantClient.h"

#include <QTcpSocket>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

namespace Ui {
class WidgetTCPAssistantClient;
};

QT_END_NAMESPACE

class TCPAssistantClient : public QWidget
{
    Q_OBJECT

public:
    TCPAssistantClient(QWidget *parent = nullptr);
    ~TCPAssistantClient();

protected slots:
    void sl_connectHandler();
    void sl_disconnectHandler();
    void sl_clientSendHandler();
    void sl_connectedMsg();
    void sl_socketErrorMsg(QAbstractSocket::SocketError socketError);
    void sl_readDataFromServer();
    void sl_disconnectedMsg();

private:
    Ui::WidgetTCPAssistantClient *ui;
    QTcpSocket                   *m_pSocket;
};
