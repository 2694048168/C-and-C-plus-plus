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
sudo pacman -S indicator-sysmonitor

# 找到并打开 System Monitor Indicator 进行设置开机自启动和显示指定状态栏信息

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
sudo pacman -S goland # Go IDE
sudo pacman -S visual-studio-code-bin # vscode
sudo pacman -S code # vscode
sudo pacman -S qtcreator # 一款QT开发软件
sudo pacman -S gitkraken # Git 管理工具

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