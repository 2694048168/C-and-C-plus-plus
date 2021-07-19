# Unbuntu OS install
```shell
# 1. 更改软件镜像源
# 2. 更新软件源索引列表
sudo apt update
# 3. 添加 root 密码
sudo passwd
# 4. 设置终端大小和透明度 100 X 30
# 5. FTP 文件传输工具 FileZilla
# https://filezilla-project.org/
sudo apt update && apt install vsftpd
sudo gedit /etc/vsftpd.conf
# 修改 ftp 配置文件，允许写入, 然后重启 OS
# write_enable=YES
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
```

- VSCode
```shell
sudo apt update
# https://code.visualstudio.com/download
sudo apt install code_1.58.0-1625728071_amd64.deb
sudo dpkg -i code_1.58.0-1625728071_amd64.deb
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