/**
 * @file 04_fund_widget.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 窗口类的基类QWidget, 对话框基类QDialog, 带菜单栏工具栏状态栏的QMainWindow, 
 * 消息对话框QMessageBox, 文件对话框QFileDialog, 字体对话框QFontDialog, 
 * 颜色对话框QColorDialog, 输入型对话框QInputDialog, 
 * 进度条对话框QProgressDialog, 资源文件
 * 
 * ====1. QWidget
 * QWidget类是所有窗口类的父类(控件类是也属于窗口类), 并且QWidget类的父类的QObject, 
 * *也就意味着所有的窗口类对象只要指定了父对象, 都可以实现内存资源的自动回收.
1.1 设置父对象
// 构造函数
QWidget::QWidget(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

// 公共成员函数
// 给当前窗口设置父对象
void QWidget::setParent(QWidget *parent);
void QWidget::setParent(QWidget *parent, Qt::WindowFlags f);
// 获取当前窗口的父对象, 没有父对象返回 nullptr
QWidget *QWidget::parentWidget() const;
// ---------------------------------------
1.2 窗口位置
//------------- 窗口位置 -------------
// 得到相对于当前窗口父窗口的几何信息, 边框也被计算在内
QRect QWidget::frameGeometry() const;
// 得到相对于当前窗口父窗口的几何信息, 不包括边框
const QRect &geometry() const;
// 设置当前窗口的几何信息(位置和尺寸信息), 不包括边框
void setGeometry(int x, int y, int w, int h);
void setGeometry(const QRect &);
    
// 移动窗口, 重新设置窗口的位置
void move(int x, int y);
void move(const QPoint &);
// ---------------------------------------
// 获取当前窗口的位置信息
void MainWindow::on_positionBtn_clicked()
{
    QRect rect = this->frameGeometry();
    qDebug() << "左上角: " << rect.topLeft()
             << "右上角: " << rect.topRight()
             << "左下角: " << rect.bottomLeft()
             << "右下角: " << rect.bottomRight()
             << "宽度: " << rect.width()
             << "高度: " << rect.height();
}

// 重新设置当前窗口的位置以及宽度, 高度
void MainWindow::on_geometryBtn_clicked()
{
    int x = 100 + rand() % 500;
    int y = 100 + rand() % 500;
    int width = this->width() + 10;
    int height = this->height() + 10;
    setGeometry(x, y, width, height);
}

// 通过 move() 方法移动窗口
void MainWindow::on_moveBtn_clicked()
{
    QRect rect = this->frameGeometry();
    move(rect.topLeft() + QPoint(10, 20));
}
// ---------------------------------------
1.3 窗口尺寸
//------------- 窗口尺寸 -------------
// 获取当前窗口的尺寸信息
QSize size() const
// 重新设置窗口的尺寸信息
void resize(int w, int h);
void resize(const QSize &);
// 获取当前窗口的最大尺寸信息
QSize maximumSize() const;
// 获取当前窗口的最小尺寸信息
QSize minimumSize() const;
// 设置当前窗口固定的尺寸信息
void QWidget::setFixedSize(const QSize &s);
void QWidget::setFixedSize(int w, int h);
// 设置当前窗口的最大尺寸信息
void setMaximumSize(const QSize &);
// ---------------------------------------
1.4 窗口标题和图标
//------------- 窗口图标 -------------
// 得到当前窗口的图标
QIcon windowIcon() const;
// 构造图标对象, 参数为图片的路径
QIcon::QIcon(const QString &fileName);
// 设置当前窗口的图标
void setWindowIcon(const QIcon &icon);

//------------- 窗口标题 -------------
// 得到当前窗口的标题
QString windowTitle() const;
// 设置当前窗口的标题
void setWindowTitle(const QString &);
// ---------------------------------------
 * ====2. QDialog
 * ====3. QDialog的子类: QMessageBox
 * ====3. QDialog的子类: QFileDialog,选择要打开/保存的文件或者目录
通用参数:
  - parent: 当前对话框窗口的父对象也就是父窗口
  - caption: 当前对话框窗口的标题
  - dir: 当前对话框窗口打开的默认目录
  - options: 当前对话框窗口的一些可选项,枚举类型, 一般不需要进行设置, 使用默认值即可
  - filter: 过滤器, 在对话框中只显示满足条件的文件, 可以指定多个过滤器, 使用 ;; 分隔
    - 样式举例: 
	- Images (*.png *.jpg)
	- Images (*.png *.jpg);;Text files (*.txt)
  - selectedFilter: 如果指定了多个过滤器, 通过该参数指定默认使用哪一个, 不指定默认使用第一个过滤器

// 打开一个目录, 得到这个目录的绝对路径
[static] QString QFileDialog::getExistingDirectory(
                  QWidget *parent = nullptr, 
                  const QString &caption = QString(), 
                  const QString &dir = QString(), 
                  QFileDialog::Options options = ShowDirsOnly);

// 打开一个文件, 得到这个文件的绝对路径
[static] QString QFileDialog::getOpenFileName(
    	          QWidget *parent = nullptr, 
    		  const QString &caption = QString(), 
                  const QString &dir = QString(), 
                  const QString &filter = QString(), 
                  QString *selectedFilter = nullptr, 
                  QFileDialog::Options options = Options());

// 打开多个文件, 得到这多个文件的绝对路径
[static] QStringList QFileDialog::getOpenFileNames(
    	          QWidget *parent = nullptr, 
                  const QString &caption = QString(), 
                  const QString &dir = QString(), 
                  const QString &filter = QString(), 
                  QString *selectedFilter = nullptr, 
                  QFileDialog::Options options = Options());

// 打开一个目录, 使用这个目录来保存指定的文件
[static] QString QFileDialog::getSaveFileName(
    		  QWidget *parent = nullptr, 
                  const QString &caption = QString(), 
                  const QString &dir = QString(), 
                  const QString &filter = QString(), 
                  QString *selectedFilter = nullptr, 
                  QFileDialog::Options options = Options());
 * ----------------------------------------------------------
 * ?QFontDialog类是QDialog的子类,进行字体属性设置的对话框窗口,调用这个类的静态成员函数
 * ?QColorDialog类是QDialog的子类, 得到一个选择颜色的对话框窗口, 调用这个类的静态成员函数
 * ?QInputDialog类是QDialog的子类, 得到一个输入对话框窗口, 调用这个类的静态成员函数
 * =====1. double QInputDialog::getDouble();
 * =====2. int QInputDialog::getInt();
 * =====3. QString QInputDialog::getItem();
 * =====4. QString QInputDialog::getMultiLineText();
 * =====5. QString QInputDialog::getText();
 * ?QProgressDialog类是QDialog的子类, 得到一个带进度条的对话框窗口, 
 * =====这种类型的对话框窗口一般常用于文件拷贝、数据传输等实时交互的场景中.
 * 
 * ====4. QMainWindow
 * ?QMainWindow是标准基础窗口中结构最复杂的窗口, 其组成如下:
 * 1. 提供了菜单栏, 工具栏, 状态栏, 停靠窗口;
 * 2. 菜单栏: 只能有一个, 位于窗口的最上方;
 * 3. 工具栏: 可以有多个, 默认提供了一个, 窗口的上下左右都可以停靠;
 * 4. 状态栏: 只能有一个, 位于窗口最下方;
 * 5. 停靠窗口: 可以有多个, 默认没有提供, 窗口的上下左右都可以停靠;
 * 
 * ====5. 资源文件 .qrc
资源文件顾名思义就是一个存储资源的文件,
在Qt中引入资源文件好处在于他能提高应用程序的部署效率并且减少一些错误的发生.
**在程序编译过程中,添加到资源文件中的文件也会以二进制的形式被打包到可执行程序中,
这样这些资源就永远和可执行程序捆绑到一起了,不会出现加载资源却找不到的问题.
虽然资源文件优势很明显,但是它也不是万能的,资源文件中一般添加的都是比较小的资源,比如图片,配置文件,MP3等,
如果是类似视频这类比较大的文件就不适合放到资源文件中了.
 * ?一个资源文件中可以添加多个前缀, 前缀就是添加的资源在资源文件中的路径, 前缀根据实际需求制定即可, 路径以 / 开头
 * 
 */

#include <QApplication> // 应用程序类头文件
#include <QColorDialog>
#include <QFontDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <iostream>

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";

    // 方式1
    // bool  ok;
    // QFont ft = QFontDialog::getFont(&ok, QFont("微软雅黑", 12, QFont::Bold), nullptr, "选择字体");

    // 方式2
    QFont ft = QFontDialog::getFont(NULL);

    // ==============================================
    QColor color = QColorDialog::getColor();
    // QBrush   brush(color);

    QStringList items;
    items << "苹果" << "橙子" << "橘子" << "葡萄" << "香蕉" << "哈密瓜";
    QString item = QInputDialog::getItem(nullptr, "请选择你喜欢的水果", "你最喜欢的水果:", items, 1, false);
    QMessageBox::information(nullptr, "水果", "您最喜欢的水果是: " + item);

    // 1. 创建进度条对话框窗口对象
    QProgressDialog *progress = new QProgressDialog("正在拷贝数据...", "取消拷贝", 0, 100, nullptr);
    // 2. 初始化并显示进度条窗口
    progress->setWindowTitle("请稍后");
    progress->setWindowModality(Qt::WindowModal);
    progress->show();
    // 3. 更新进度条
    // static int value = 0;
    // QTimer    *timer = new QTimer;
    // connect(timer, &QTimer::timeout, nullptr,
    //         [=]()
    //         {
    //             progress->setValue(value);
    //             value++;
    //             // 当value > 最大值的时候
    //             if (value > progress->maximum())
    //             {
    //                 timer->stop();
    //                 value = 0;
    //                 delete progress;
    //                 delete timer;
    //             }
    //         });

    // connect(progress, &QProgressDialog::canceled, nullptr,
    //         [=]()
    //         {
    //             timer->stop();
    //             value = 0;
    //             delete progress;
    //             delete timer;
    //         });

    // timer->start(50);

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
