# Manjaro OS install

- Install wiki

https://manjaro.org/

https://manjaro.org.cn/


- 软件源

```shell
# 添加国内源
# 注意：使用顺序，从上往下优先级越来越低，越靠上，优先级越高
# 1. 添加之前首先备份原文件
sudo cp /etc/pacman.d/mirrorlist /etc/pacman.d/mirrorlist.backup
# 2. 编辑 /etc/pacman.d/mirrorlist 配置文件
sudo vim /etc/pacman.d/mirrorlist
# 3. 添加 manjaro 稳定源
## 中科大
Server = https://mirrors.ustc.edu.cn/manjaro/stable/$repo/$arch

##  清华大学
Server = https://mirrors.tuna.tsinghua.edu.cn/manjaro/stable/$repo/$arch

## 上海交通大学
Server = https://mirrors.sjtug.sjtu.edu.cn/manjaro/stable/$repo/$arch

## 浙江大学
Server = https://mirrors.zju.edu.cn/manjaro/stable/$repo/$arch


# 4. 添加 archlinux 源
# 清华大学
Server = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch
## 163
Server = http://mirrors.163.com/archlinux/$repo/os/$arch
## aliyun
Server = http://mirrors.aliyun.com/archlinux/$repo/os/$arch


# 中文社区仓库
# 1. 备份原文件
sudo cp /etc/pacman.conf /etc/pacman.conf.backup
# 2. 编辑 /etc/pacman.conf 配置文件 color 打开注释
sudo vim /etc/pacman.conf
# 3. 添加源 [注意：只能添加一个]
[archlinuxcn]
# The Chinese Arch Linux communities packages.
# SigLevel = Optional TrustedOnly
SigLevel = Optional TrustAll
# 官方源
Server   = http://repo.archlinuxcn.org/$arch
# 163源
Server = http://mirrors.163.com/archlinux-cn/$arch
# 清华大学
Server = https://mirrors.tuna.tsinghua.edu.cn/archlinuxcn/$arch


# 添加 AUR 源 yaourt 用户
# 1. 添加之前首先备份原文件
sudo cp /etc/yaourtrc /etc/yaourtrc.backup
# 2. 修改 /etc/yaourtrc 配置文件
sudo vim /etc/yaourtrc
# 3. 去掉 # AURURL 的注释,并修改
AURURL=“https://aur.tuna.tsinghua.edu.cn”

# yay 用户
# 1. 执行以下命令修改 aururl :
yay --aururl “https://aur.tuna.tsinghua.edu.cn” --save
# 2. 修改的配置文件
sudo vim ~/.config/yay/config.json
# 3. 查看配置
yay -P -g

# 手动更改源排名
sudo pacman-mirrors -i -c China -m rank

```

- pacman 管理

```shell
# 1. 安装软件
sudo pacman -S
sudo pacman -S code
sudo pacman -Sy      # 更新软件源
sudo pacman -Syy     # 强制更新软件源 (可能最近一次更新过一次)
sudo pacman -Su      # 更新软件
sudo pacman -Syu     # 更新软件源后更新已经安装的软件
sudo pacman -Syyu    # 强制更新软件源并更新软件
sudo pacman -Ss vim  # 在软件源里面搜索软件名，支持正则表达式
sudo pacman -Sc      # 删除系统每次安装软件后缓存的软件包

# 2. 卸载软件
sudo pacman -R
sudo pacman -R vim  # 卸载软件
sudo pacman -Rs vim  # 卸载软件及其依赖
sudo pacman -Rns vim  # 卸载软件及其依赖，同时删除该软件的全局配置文件(用户的配置文件不会删除)

# 3. 查询已经安装的软件
sudo pacman -Q
sudo pacman -Q | wc -l  # 统计已经安装的软件数量
sudo pacman -Qe  # 查询用户自己安装的软件情况(除去系统的)
sudo pacman -Qe | wc -l  # 统计用户自己安装的软件情况(除去系统的)
sudo pacman -Qeq  # 软件的版本号不显示，用于重定向用户安装的软件到文件
sudo pacman -Qeq > manjaro_pacages.txt # 软件的版本号不显示，用于重定向用户安装的软件到文件
sudo pacman -Qs vim  # 查询本地已经安装的某一个软件信息
sudo pacman -Qdt  # 查询不再被依赖的孤立的软件依赖包
sudo pacman -Qdtq  # 查询不再被依赖的孤立的软件依赖包,不显示版本号

# 灵活利用，并配合 shell 进行操作
# 删除不再被依赖的所有依赖包
sudo pacman -R $(pacman -Qdtq)

```

- 常用操作

```shell
# 选择国内镜像源 - 排名出界面进行选择
sudo pacman-mirrors -i -c China -m rank

# 更新镜像源索引
sudo pacman -Syy

# 更新整个系统
sudo pacman -Syu

# 下面要编辑 pacman.conf，添加 archlinuxcn 源。
# archlinuxcn 是一个由 Arch Linux 中文社区驱动的非官方用户仓库
sudo vi /etc/pacman.conf
# 在文件最后添加
[archlinuxcn]
SigLevel = Optional TrustedOnly
Server = https://mirrors.tuna.tsinghua.edu.cn/archlinuxcn/$arch

# 更新软件源索引并安装
sudo pacman -Syy
sudo pacman -S archlinux-keyring

# 有了 archlinuxcn 源就可以安装输入法了。 
# 安装时不能直接安装最后一个包，靠依赖安装 fcitx，这样会导致 fcitx 版本缺失。 
# 须使用下面顺序安装，其中 fcitx-im 使用默认选项，安装每个版本的 fcitx
# 使用 yay 安装最简单，百度一下就知道了
sudo pacman -S fcitx
sudo pacman -S fcitx-im
sudo pacman -S fcitx-configtool
sudo pacman -S fcitx-googlepinyin

# 安装好后编辑用户，使在每个环境下都使用 fcitx。编辑 ~/.xprofile 文件
vi ~/.xprofile
# 添加内容
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS="@im=fcitx"

# 设置结束后重启即可在fcitx中找到输入法了

# 设置终端打开快捷键 gnome 模拟终端
# 添加自定义快捷键：设置> 设备> 键盘> ＋>添加快捷键
gnome-terminal 

# Manjaro 支持 ftp 服务 (一般用不到，桌面用户)
sudo pacman -S vsftpd
sudo nano /etc/vsftpd.conf
# 通过根据需要删除任何前面的哈希号取消注释
anonymous_enable=NO
local_enable=YES
write_enable=YES
local_umask=022

# 完成后保存并关闭文件，启动 vsftpd 和 设置开机自启
sudo systemctl enable vsftpd
sudo systemctl start vsftpd

```

- Manjaro 实时显示内存和CPU占用以及网速的情况
```shell
# Linus-Torvalds-Talk-is-cheap-Show-me-the-code
# https://quotefancy.com/quote/1445782/Linus-Torvalds-Talk-is-cheap-Show-me-the-code

# https://github.com/fossfreedom/indicator-sysmonitor

sudo pacman -S indicator-sysmonitor

# 找到并打开 System Monitor Indicator 进行设置开机自启动和显示指定状态栏信息
CPU {cpu}  {cputemp}   |  GPU {nvgpu}  {nvgputemp}  |  MEM {mem}  |  SWAP {swap}  |  Net Speed Compact {netcomp}  |  Total Net Speed {totalnet}

# 雷神终端 - Guake
# 可以按 F12 下拉一个终端，用起来会很方便
sudo pacman -S guake

# 终端 tmux
# Tmux 是一个终端复用器（terminal multiplexer），非常有用，属于常用的开发工具
# https://github.com/tmux/tmux/wiki
# https://learnxinyminutes.com/docs/zh-cn/tmux-cn/
# https://ovirgo.com/tmux.html
# 简介：https://www.ruanyifeng.com/blog/2019/10/tmux.html
sudo pacman -S tmux

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


```


- 基本开发配置

```shell
# 配置 zsh 和 oh-my-zsh
sudo pacman -Syu zsh
# 查看 shell 版本 切换默认使用 zsh
echo $SHELL
zsh --version
which zsh
chsh -s $(which zsh)

# 安装 oh-my-zsh, 配置 zsh
# https://ohmyz.sh/
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
# sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# 下载 zsh-syntax-highlighting 语法高亮插件
# git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-syntax-highlighting
git clone https://hub.fastgit.org/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-syntax-highlighting

# 下载 zsh-autosuggestions 自动提示插件
# git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-autosuggestions
git clone https://hub.fastgit.org/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh}/plugins/zsh-autosuggestions

# 配置 .zshrc文件 更换默认主题为： ZSH_THEME=alanpeabody
# https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
nano ~/.zshrc
# 添加内容
plugins=(git zsh-syntax-highlighting zsh-autosuggestions)

# 配置生效
source ~/.zshrc
. ~/.zshrc


# sudo apt install build-essential (包含一些的基本工具 gcc make g++ 等等)
sudo pacman -Sy base-devel

# install VSCode
sudo pacman -S code

# install Git
sudo pacman -S git cmake

# deepin 系的软件
yay -s deepin-wine-wechat #微信
yay -S deepin.com.thunderspeed #迅雷

# 开发软件
sudo pacman -S jdk8-openjdk
sudo pacman -S cmake
sudo pacman -S clang
sudo pacman -S vim
sudo pacman -S pycharm-professional # Python IDE
sudo pacman -S pycharm-community-edition # Python IDE
sudo pacman -S goland # Go IDE
sudo pacman -S visual-studio-code-bin # vscode
sudo pacman -S code # vscode
sudo pacman -S qtcreator # 一款QT开发软件

# 办公软件
sudo pacman -S google-chrome
sudo pacman -S foxitreader # pdf 阅读
sudo pacman -S wps-office
yay -S typora # markdown 编辑
yay -S xmind #思维导图

# 娱乐软件
sudo pacman -S netease-cloud-music #网易云音乐

# 下载软件
sudo pacman -S filezilla  # FTP/SFTP

# 终端
sudo pacman -S screenfetch # 终端打印出你的系统信息，screenfetch -A 'Arch Linux'
sudo pacman -S net-tools # 这样可以使用 ifconfig 和 netstat
yay -S tree #以树状图列出目录的内容

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
```

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

```


- 配置 python 开发环境
```shell
which python
which python3
python --version
python3 --version

# 安装 pip 包管理工具
# python-packageName
# python2-packaName
sudo pacman -Syy python-pip

# 设置全局默认pypi国内镜像源地址，只需要一个即可
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 创建 python 虚拟环境
# https://docs.python.org/zh-cn/3/tutorial/venv.html
# https://docs.python.org/zh-cn/3/library/venv.html#module-venv
# --------------------------------------------------------------
# venv 模块支持使用自己的站点目录创建轻量级“虚拟环境”，可选择与系统站点目录隔离。
# 每个虚拟环境都有自己的 Python 二进制文件（与用于创建此环境的二进制文件的版本相匹配），
# 并且可以在其站点目录中拥有自己独立的已安装 Python 软件包集。
# 这样可能会导致升级后，虚拟环境中的 python 解析器版本依赖于系统的，出现不兼容问题
# 建议使用 virtualenv 包来进行虚拟环境的隔离
# --------------------------------------------------------------
python3 -m venv ~/virtual_env/pytorch

# 激活环境 bash
# 这个脚本是为 bash shell编写的。如果使用 csh 或 fish shell，你应该改用 activate.csh 或 activate.fish 脚本。）
source tutor~/virtual_env/pytorch/bin/activate

# 退出环境
deactivate

```