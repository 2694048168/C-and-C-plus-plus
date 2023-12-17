#pragma once

#include <QThread>
#include <QImage>
#include <string>


class ThreadShowImage : public QThread
{
    Q_OBJECT

public:
    explicit ThreadShowImage(QObject* parent = nullptr);

protected:
    void run() override;

signals:
    // 自定义信号, 传递数据
    void currentImg(QImage img);

private:
    std::string m_imgPath;
};
