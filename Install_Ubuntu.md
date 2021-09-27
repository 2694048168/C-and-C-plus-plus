# Ubuntu OS install
```shell
# 1. 更改软件镜像源
# 2. 更新软件源索引列表
sudo apt update
# 3. 添加 root 密码
sudo passwd
# ssh 服务器
sudo apt update
sudo apt install openssh-server
sudo systemctl status ssh
# 防火墙？
sudo ufw allow ssh
# 4. 设置终端大小和透明度 100 X 30
# 5. FTP 文件传输工具 FileZilla
# https://filezilla-project.org/
sudo apt update && apt install vsftpd
sudo gedit /etc/vsftpd.conf
# 修改 ftp 配置文件，允许写入, 然后重启 OS
# write_enable=YES
```

# Linux Ubuntu First To Do

```shell
# 1. 更新软件镜像源
# 2. 登录 Livepatch 以便获取补丁更新
# 3. 确定显卡等硬件的驱动安装以及更新
# 4. 终端 shell 设置 zsh 以及美化
# 5. Ubutu20.04 美化：https://zhuanlan.zhihu.com/p/176977192


# apt 管理工具
sudo apt update  # 获取最新的软件包列表
sudo apt install # 安装软件
sudo apt remove  # 卸载软件
sudo apt clean   # 清理下载缓存的安装包

# 利用 dpkg 管理 deb 格式
sudo dpkg -i xxx.deb  # 安装 deb 格式的软件
sudo dpkg -r xxx      # 卸载 deb 格式安装的软件
```

# Terminal Shell

- Shell

```shell
# 检查当前可用的 shell
cat /etc/shells

# 查看当前使用的shell
echo $SHELL
# 安装 zsh shell
sudo apt install zsh -y

# 查看 shell 版本 切换默认使用 zsh
zsh --version
chsh -s $(which zsh)
chsh -s $(which bash) root

# 安装 oh-my-zsh, 配置 zsh
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
# sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# 下载 zsh-syntax-highlighting 语法高亮插件
# git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-syntax-highlighting
git clone https://hub.fastgit.org/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-syntax-highlighting

# 下载 zsh-autosuggestions 自动提示插件
# git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-autosuggestions
git clone https://hub.fastgit.org/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-autosuggestions

# 配置 .zshrc文件 更换默认主题为： agnoster
sudo apt-get install fonts-powerline  # agnoster 主题需要以来字体
vim ~/.zshrc
gedit ~/.zshrc
# 添加内容
plugins=(git zsh-syntax-highlighting zsh-autosuggestions)

# 配置生效
source ~/.zshrc

# xwininfo，记下输出的最后一行
# -geometry 110x30+246-59
gnome-terminal --geometry 110x30+246-59 

# 终端字符画
sudo apt install lolcat figlet
# figlet 的作用是画字符画
echo "Hello World" | figlet
# lolcat 的作用是把字符串变成炫彩的
echo "Hello World" | figlet | lolcat
# 想要显示的内容加到 .zshrc 的最后一行
# -a -d 1 参数的作用是让这个炫彩的动画花一秒钟慢慢出来
gedit ~/.zshrc
echo "Hello World" | figlet | lolcat -a -d 1
echo "Hello World" | figlet | lolcat

```

# Vim or NeoVim

```shell
# 0. vim 四种模式：普通模式；输入模式；命令模式；可视模式
# 1. 普通模式：用于浏览文件，执行复制、粘贴、删除之类的操作
# 2. 输入模式：用于编辑或者改变文件内容
# 3. 命令模式：用于控制文件的保存，退出等操作
# 4. 可视模式：用于可视化选择文件内容

# 普通模式下
# 5. 光标移动 hjkl
# 6. num+hjkl, 表示光标移动行数，列数
# 7. w 跳转下一个单词的开头
# 8. b 跳转前一个单词的开头
# 9. gg，跳转文件最上方
# 10. G，跳转文件最下方
# 11. ctrl+u，向上翻页
# 12. ctrl+d，向下翻页
# 13. y(yank)，表示复制，结合操作，yaw(Yank All Words)复制整个单词, y4j(复制当前向下4行内容)
# 14. d(delete)，表示删除，结合操作，daw(Yank All Words)删除整个单词, d4j(删除当前向下4行内容)
# 15. u(undo)，表示撤销
# 16. c(change)，表示改变文件内容方式进行输入模式，caw(change all words)删除整个单词并进入输入模式，cc删除当前行并进入输入模式，c4j 删除当前向下4行并进入输入模式

sudo apt install vim
vim file

# https://vimawesome.com/
# https://github.com/junegunn/vim-plug

# 利用 vscode 远程连接 ssh 完成代码的编辑和编译
# vscode 插件 Remote-SSH
ssh weili@192.168.128.128
```

# tools
- SogouPinyin
```shell
# 20.04.2.0 (amd64, Desktop LiveDVD)
# 这个 ubuntu20 的版本才支持，其他 20 版本由于取消 QT，不支持
sudo apt update
sudo apt install fcitx -y
# 设置里面配置一下 fcitx
# https://pinyin.sogou.com/linux/
sudo dpkg -i sogoupinyin_2.3.1.0112_amd64.deb
# 有问题执行下面命令
sudo apt install -f
# 注销一下就可以使用了
# fcitx-configtool 打开配置输入
fcitx-configtool
```

- ubuntu 实时显示内存和CPU占用以及网速的情况
```shell
# https://github.com/fossfreedom/indicator-sysmonitor
# 添加软件源的命令
sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor && sudo apt update
# 如果需要删除该软件源可以使用
sudo add-apt-repository -r ppa:fossfreedom/indicator-sysmonitor

# 安装 indicator-sysmonitor
sudo apt install indicator-sysmonitor -y

# 找到并打开 System Monitor Indicator 进行设置开机自启动和显示指定状态栏信息
nohup indicator-sysmonitor &

CPU {cpu}  {cputemp}   |  GPU {nvgpu}  {nvgputemp}  |  MEM {mem}  |  SWAP {swap}  |  Net Speed Compact {netcomp}  |  Total Net Speed {totalnet}

# 另一种方法就是使用 美化插件 NetSpeed 扩展

# 雷神终端 - Guake
# 可以按 F12 下拉一个终端，用起来会很方便
sudo apt install guake

# 终端 tmux
# Tmux 是一个终端复用器（terminal multiplexer），非常有用，属于常用的开发工具
# https://github.com/tmux/tmux/wiki
# https://learnxinyminutes.com/docs/zh-cn/tmux-cn/
# https://ovirgo.com/tmux.html
# 简介：https://www.ruanyifeng.com/blog/2019/10/tmux.html
sudo apt install tmux


# 进入 tmux 终端
tmux

# 退出 tmux 终端
Ctrl + d

# 帮助命令的快捷键是 Ctrl+b ?
# 它的用法是，在 Tmux 窗口中，先按下 Ctrl+b，再按下 ?，就会显示帮助信息。
# 然后，按下 ESC 键或q键，就可以退出帮助

# 划分左右两个窗格 先按下激活键 Ctrl+b, 再按下 shift+”
Ctrl+b %

Ctrl+b "：划分上下两个窗格

Ctrl+b <arrow key>：光标切换到其他窗格。<arrow key>是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键↓。
Ctrl+b ;：光标切换到上一个窗格。
Ctrl+b o：光标切换到下一个窗格。
Ctrl+b {：当前窗格与上一个窗格交换位置。
Ctrl+b }：当前窗格与下一个窗格交换位置。
Ctrl+b Ctrl+o：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
Ctrl+b Alt+o：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
Ctrl+b x：关闭当前窗格。
Ctrl+b !：将当前窗格拆分为一个独立窗口。
Ctrl+b z：当前窗格全屏显示，再使用一次会变回原来大小。
Ctrl+b Ctrl+<arrow key>：按箭头方向调整窗格大小。
Ctrl+b q：显示窗格编号。


## 配置 conda 和 pycharm 

```shell
# 1. download Miniconda for Linux
# https://docs.conda.io/en/latest/miniconda.html#linux-installers

# 2. 添加脚本的执行权限
chmod a+x Miniconda3-latest-Linux-x86_64.sh

# 3. 运行安装脚本
./Miniconda3-latest-Linux-x86_64.sh
# 4. 接受协议，确定安装路径，conda init 选择 no

# 5. 添加 conda 到环境变量中
# vi ~/.zshrc
gedit ~/.zshrc
export PATH=/home/weili/miniconda3/bin:$PATH  # 把 anaconda 安装的 bin 目录写入配置文件

# 6. 生效环境配置，测试 conda
source ~/.zshrc
conda --version
pip --version
python3 --version
which python3
which pip
which conda

# ----------------------------------
# 1. download pycharm-community
# https://www.jetbrains.com/zh-cn/pycharm/download/#section=linux

# 2. 解压 tar 包
tar -zxvf pycharm-community-2021.2.1.tar.gz

# 3. 重命名加压包后的文件夹名称
mv pycharm-community-2021.2.1 pycharm-community

# 4. 进入安装路径的 bin 执行启动脚本
cd pycharm-community/bin
./pycharm.sh

# 5. 启动界面的左下角找到 setting 的图标
# 选择 create Desttop Entry，创建桌面快捷入口和菜单入口，方便启动

# ----------------------------------
# 1. 利用 conda 进行虚拟环境的隔离
# 创建虚拟环境
conda create -n SR_pytorch_1_8_2 python=3.8
# 查看已有的环境名称
conda info -e
# 进入虚拟环境
conda activate SR_pytorch_1_8_2
# 退出环境
conda deactivate or source deactivate
# 删除环境
conda remove --name SR_pytorch_1_8_2 --all

# 2. 利用 pip 进行 python 软件包的管理
# 配置 pip 软件镜像源, 永久设置全局 pypi 镜像源命令
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 3. 利用 pycharm 进行源代码的调试
# 从 pycharm 的终端进行, 可以直接定位到虚拟环境中, pip 管理具体环境中的包

# ----------------------------------
# 安装 向日葵 进行远程控制
# 1. 下载选择 图形版本 ，命令行版本不支持发起远程控制(只能被控制)
# https://sunlogin.oray.com/download

# 2. 使用 sudo 进行安装
sudo dpkg -i sunloginclient-11.0.0.36662-amd64.deb

# 3. 菜单栏启动即可使用

```

- gcc/g++/make
```shell
sudo apt update
sudo apt upgrade
sudo apt install build-essential -y
```

- cmake/git/python3
```shell
sudo apt update
sudo apt install cmake git -y

# Python3 系统默认安装
sudo apt install python3.8

git config --global user.email "weili_yzzcq@163.com"
git config --global user.name "Wei Li"
# GitHub上 传文件不能超过1 00M 的解决办法, 修改为 500M
git config http.postBuffer 524288000
git config -l
```

- VSCode
```shell
sudo apt update
# https://code.visualstudio.com/download
sudo apt install code_1.58.0-1625728071_amd64.deb
sudo dpkg -i code_1.58.0-1625728071_amd64.deb

cpp.json 配置智能指令 cpp

{
	// Place your snippets for cpp here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	// "Print to console": {
	// 	"prefix": "log",
	// 	"body": [
	// 		"console.log('$1');",
	// 		"$2"
	// 	],
	// 	"description": "Log output to console"
	// }

	"Print to conaole":{
    "prefix": "cpp",    //在新建立的页面中输入C++就会有智能提示，Tab就自动生成好了
    "body": [
			  "/**",
			  " * @File    : ${TM_FILENAME}",
		      " * @Brief   : $1",
		      " * @Link    : $2",
			  " * @Author  : Wei Li",
			  " * @Date    : ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}",
			  "*/",
				"", //空行
        "#include <iostream>",
        "", //空行
        "int main(int argc, char** argv)",   //main()函数
        "{",
        "    $0",    //最终光标会在这里等待输入
        "    return 0;", //结束
        "}",
        "",
    ],
}
}

vscode 插件
c/c++ IntelliSense
Guides
Path Intellisense
Python
Python Docstring Generate
CMake Tools
```

- PyTorch/TensorFlow
```shell
python3 --version
# install pip Python 包管理工具
sudo apt install python3-pip -y

# 配置 pip 软件镜像源, 永久设置全局 pypi 镜像源命令
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

mkdir python_venv
cd python_venv
# install python3-venv 虚拟环境管理, 根据提示安装即可
# https://docs.python.org/zh-cn/3.8/library/venv.html?highlight=venv
sudo apt install python3.8-venv
python3 -m venv pytorch

# 激活虚拟环境
source pytorch/bin/activate
# 退出虚拟环境
deactivate

# install PyTorch
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 需要 NVIDIA 驱动程序以及硬件显卡支持
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install PyTorch
# 需要 NVIDIA 驱动程序以及硬件显卡支持
pip install tensorflow==2.5.0

pip install tensorflow-cpu==2.5.0
```
