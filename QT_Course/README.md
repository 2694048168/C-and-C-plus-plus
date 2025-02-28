## Modern C++ and QT

> Modern C++ and QT creating software applications, from planning and design to development, testing, and future-proofing your products.

- [Qt官方下载地址](https://download.qt.io/archive/qt/)
- [VSCode](https://code.visualstudio.com/download)
- [CMake](https://cmake.org/download/)

```shell
# install the QT5 and QT6
# - Source: Qt源码, 另外Qt的一些模块运行需要的驱动没有提供现成的动态库需要自己编译, 建议安装
# - Qt Charts: 用于绘制统计数据对应的图表, 比如: 折线图/曲线图等

# VSCode + CMake + Qt Official extension
# MSVC无法自动生成 compile_commands.json

# VSCode + CMake + Qt Support extension + cland
# 采用 Ninja 构建编译 自动生成 compile_commands.json
cmake -S . -B build -G "Ninja"
cmake --build build

bash build.sh

# C:\Qt\5.15.2\mingw81_64\bin
windeployqt 00_dev_env.exe
```

### Qt内存管理机制

> Qt中创建对象的时候会提供一个**Parent对象指针**(可以查看类的构造函数),这个parent到底是干什么的? QObject是以**对象树**的形式组织起来的, 当创建一个QObject对象时, 会看到QObject的构造函数接收一个QObject指针作为参数, 这个参数就是 parent, 也就是父对象指针. 这相当于在创建QObject对象时, 可以提供一个其父对象, 创建的这个QObject对象会自动添加到其父对象的children()列表; 当父对象析构的时候, 这个列表中的所有对象也会被析构.(注意,**这里的父对象并不是继承意义上的父类**). QWidget是能够在屏幕上显示的一切组件的父类, QWidget继承自QObject, 因此也继承了这种对象树关系. 一个孩子自动地成为父组件的一个子组件, 因此它会显示在父组件的坐标系统中, 被父组件的边界剪裁. 例如当用户关闭一个对话框的时候, 应用程序将其删除, 那么希望属于这个对话框的按钮、图标等应该一起被删除, 事实就是如此, 因为这些都是对话框的子组件. Qt 引入对象树的概念, 在一定程度上解决了**内存问题**; 当一个QObject对象在堆上创建的时候, Qt 会同时为其创建一个对象树; 不过对象树中对象的顺序是没有定义的, 这意味着**销毁这些对象的顺序也是未定义的**. 任何对象树中的 QObject对象 delete 的时候,如果这个对象有 parent, 则自动将其从 parent 的children()列表中删除; 如果有孩子, 则自动 delete 每一个孩子; Qt 保证没有QObject会被 delete 两次, 这是由**析构顺序决定的**.

### Features
- Qt事件介绍与正弦曲线绘制 **18_sine_curve_drawing.cpp**
- Qt应用软件登录界面 **19_login_widget.cpp**
- Qt联合OpenGL图形渲染 **20_graph_render.cpp**
- Qt联合OpenGL图像显示 **21_image_render.cpp**

### Qt基础
- Qt认识入门 **00_dev_env.cpp**
- Qt中的基础数据类型 **01_data_type.cpp**
- Qt中的信号槽 **02_signal_slot.cpp**
- Qt定时器类QTimer **03_timer.cpp**
- Qt中的基础窗口类 **04_fund_widget.cpp**
- [Qt窗口布局](https://subingwen.cn/qt/qt-layout/) **05_layout_widget.cpp**
- 在Qt窗口中添加右键菜单 **06_right_click.cpp**

### Qt控件
- Qt中按钮类型的控件 **07_widget_btn.cpp**
- Qt中容器类型的控件 **08_widget_container.cpp**

### Qt事件
- Qt事件之事件处理器 **09_event_processor.cpp**
- Qt事件之事件分发器 **10_event_distributor.cpp**
- Qt事件之事件过滤器 **11_event_filter.cpp**

### 套接字通信
- Qt中基于TCP的套接字通信 **12_socket_TCP.cpp**

### Qt线程
- Qt中多线程的使用 **13_thread_task.cpp**
- Qt中线程池的使用 **14_thread_pool.cpp**

### 数据库
- [Qt编译MySql插件](https://subingwen.cn/qt/sql-driver/) **15_sql_test.cpp**
- Qt中的数据库操作 **16_database_test.cpp**
- Qt中的Json操作 **17_json_test.cpp**

### QT多语言支持和动态切换
- Qt 语言处理工具, lupdate、lrelease、linguist
- 利用 lupdate(Language Update)工具(*.ui/*.h/*.cpp)生成 .ts 文件, tr() function
- 利用 lrelease(Language Release)将 .ts 文件翻译成指定语言的 .qm 文件
- 启动 linguist.exe(语言处理专家), 进行翻译, Qt\5.15.2\mingw81_64\bin\linguist.exe

```C++
QTranslator *qTranslator = nullptr;

void Dialog::on_btnChinese_clicked() {
    if (nullptr == qTranslator) {
        qTranslator = new QTranslator(this);
    }
    qTranslator->load("ProjectName_zh_CN.qm");
    qApp->installTranslator(qTranslator);
    ui->retranslateUi(this);
    ui->labelText->setText(tr("这是一个QT多语言程序（中文）。"));
}

void Dialog::on_btnEnglish_clicked() {
    if (nullptr == qTranslator) {
        qTranslator = new QTranslator(this);
    }
    qTranslator->load("ProjectName_en_US.qm");
    qApp->installTranslator(qTranslator);
    ui->retranslateUi(this);
    ui->labelText->setText(tr("这是一个QT多语言程序（英文）。"));
}
```

### 打包部署
- Qt程序的发布和打包

> 将应用程序和相关的动态库打包成安装包的工具有很多,一个是 NSIS Edit,一个是 Inno Setup(这是一个免费的) Windows 安装程序制作软件，小巧、简便、精美.

- [Inno Downloads](https://jrsoftware.org/isdl.php#stable)
- [NSIS Downloads](https://nsis.sourceforge.io/Download)
