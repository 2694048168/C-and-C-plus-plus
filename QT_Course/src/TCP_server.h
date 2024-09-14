#ifndef __TCP_SERVER_H__
#define __TCP_SERVER_H__

#include <QTcpServer>
#include <QTcpSocket>
#include <QWidget> // Qt标准窗口类头文件

QT_BEGIN_NAMESPACE

// mainwindow.ui 文件中也有一个类叫 MainWindow, 将这个类放到命名空间 Ui 中
namespace Ui {
class TCPServer;
}

QT_END_NAMESPACE

class TCPServer : public QWidget
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : TCPServer(QWidget *parent = nullptr);
    ~TCPServer();

private slots:
    // void on_startServer_clicked();
    void sl_startServer_clicked();
    // void on_sendMsg_clicked();
    void sl_sendMsg_clicked();

private:
    Ui::TCPServer *ui; // 定义指针指向窗口的 UI 对象
    QTcpServer    *m_pServer;
    QTcpSocket    *m_pTcp;

    const QString m_red_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:red";

    const QString m_green_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:green";

    const QString m_grey_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:grey";

    const QString m_yellow_SheetStyle
        = "min-width: 16px; min-height: 16px;max-width:16px; max-height: 16px;border-radius: 8px;  border:1px solid "
          "black;background:yellow";
};
#endif // __TCP_SERVER_H__
