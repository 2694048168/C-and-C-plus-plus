/**
 * @file SerialAssistant.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "ui_SerialAssistant.h"

#include <QCheckBox>
#include <QLineEdit>
#include <QList>
#include <QPushButton>
#include <QSerialPort>
#include <QString>
#include <QTimer>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

namespace Ui {
class WidgetSerialAssistant;
};

QT_END_NAMESPACE

class SerialAssistant : public QWidget
{
    Q_OBJECT

public:
    SerialAssistant(QWidget *parent = nullptr);
    ~SerialAssistant();

private slots:
    void sl_OpenCloseSerial();
    void sl_SendContext();
    void sl_SerialDataReadyToRead();
    void on_checkBox_sendInTimer_clicked(bool checked);
    void sl_ClearRecvMessage();
    void sl_SaveRecvMessage();
    void on_checkBox_HexShow_clicked(bool checked);
    void on_btn_hideTable_clicked(bool checked);
    void on_btn_hideRecord_clicked(bool checked);

    void sl_command1();
    void sl_command2();
    void sl_command3();
    void sl_command4();
    void sl_command5();
    void sl_command6();
    void sl_command7();
    void sl_command8();
    void sl_command9();
    void sl_command();

    void on_checkBox_cycleSend_clicked(bool checked);
    void sl_handleCycleSend();
    void sl_handleTableReset();
    void sl_handleTableSave();
    void sl_handleTableLoad();

private:
    void initSerialPorts();
    void initConnects();
    void initWidgetStatus();

private:
    Ui::WidgetSerialAssistant *ui;
    QSerialPort               *m_pSerialPort;
    int                        m_writeBytesCounterTotal;
    int                        m_readBytesCounterTotal;
    // std::string                m_recordMsgBack;
    QString                    m_recordMsgBack;
    bool                       m_isOpen;
    QTimer                    *m_pTimerInSend;
    QTimer                    *m_pTimerSysDateTime;
    QString                    m_currentDateTimeStr;
    QList<QPushButton *>       m_pButtons;
    QList<QLineEdit *>         m_pLineEdits;
    QList<QCheckBox *>         m_pCheckBoxes;
    QTimer                    *m_pTimerCycleSend;
    int                        m_btnIndex;

    const QString m_red_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:red";

    const QString m_green_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:green";

    const QString m_gray_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:grey";

    const QString m_yellow_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:yellow";
};
