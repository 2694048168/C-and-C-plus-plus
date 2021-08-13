# Linux Bash

---------------------------

## Bash command

```shell
# 常用命令
ls
ls -lah
cd
pwd
cp
mkdir
touch
mv
df -lh
rm -rf 
cat
less
find
grep
chmod a+x xxx.sh

# 查看文件系统结构
tree / ranger
sudo apt install tree
sudo pacman -S tree


>>  追加方式 
2> | 重定向和管道
标准输出 1
标准错误输出 2

tar [选项] 打包文件名 文件
    -c 生成档案文件，创建打包文件
    -v 列出归档解档的详细过程，显示进度
    -f 指定档案文件名， f 后面一定是 tar 文件，必须放在组合选项的最后
    -x 解开档案
    -z 使用 gzip 进行压缩算法
    -j 使用 bzip2 进行压缩算法
    -C 解压到指定目录

# 压缩
tar -czvf filename.tar.gz file1.txt file2.txt file3.txt file4.txt
tar -cjvf filename.tar.bz2 file1.txt file2.txt file3.txt file4.txt
zip -r 目标文件(不需要扩展名，自动添加 .zip) 源文件

# 解压
tar -xzvf filename.tar.gz -C /home/weili/
tar -xjvf filename.tar.bz2 -C /home/weili/
unzip -d 解压后目录文件 压缩文件


# 终端命令
Tab 键补全命令和路径以及显示当前目录下的所有文件
clear
Ctrl + L 清屏快捷键
Ctrl + c 中断终端操作

Ctrl + a 移动光标到头部，行首
Ctrl + e 移动光标到尾部，行尾
Ctrl + b 光标先后（左移）一个字符
Ctrl + f 光标先前（右移）一个字符

Ctrl + u 清除光标之前的所有字符
Ctrl + k 清除光标之后的所有字符

Ctrl + p 查看之前的 shell 历史命令，键盘的向上方向键
Ctrl + n 查看之后的 shell 历史命令，键盘的向下方向键

# 选择终端进行操作
Ctrl + Shift + N 新建一个终端
Ctrl + Shift + T 再该终端中新建一个标签
Ctrl + Shift + W 关闭当前的终端标签
Ctrl + Shift + C 复制
Ctrl + Shift + C 粘贴
Ctrl + Shift + = 放大终端字体大小
Ctrl + - 缩小终端字体大小
Ctrl + 0 终端字体恢复到正常的普通大小（数字零）

# 推荐终端字体：DejaVu Sans Mono Book

# 内建命令和外部命令
# 查看 command 是内建命令还是外部命令
type [-afptP] command

# 内建命令 help
help pwd
# 外部命令 帮助信息
ls --help

# 命令帮助文档
man
1 表示可执行程序或者 shell 命令
2 表示系统调用(内核提供的函数)
3 表示库调用(程序库中的函数)
4 表示特殊文件(通常位于 /dev)
5 表示文件格式和规范 如 /etc/passwd
6 表示游戏
7 表示一些杂项(包括宏和规范)
8 表示系统管理命令(通过只针对 root 用户)
9 表示内核例程

```

---------------------------

## shell script

```shell

# 学习 shell 完成简单任务并高效的
# 对 linux 的操作更加熟悉，更深的理解   

# 确定当前用户用什么 shell
echo $SHELL   # 查看环境变量或者看 passwd 文件用户的启动命令
chsh          # 修改当前用户的 shell
        
# shell 的内建命令
# shell 的内建命令是不会启动一个进程，而是就在当前 shell 的进程下去调用一个普通函数
type shell_command

# shell 脚本的执行
# 1 直接./执行
    启动一个子进程 
    使用解释器解析脚本中的每一句命令
    使用哪个解释器，在脚本的第一句指定
    #!/bin/bash

    在子进程中cd 更换目录不会影响父进程
# 2 /bin/sh 脚本地址
    也是启动一个子进程来执行，跟第一种方法作用类似
    这种方法不需要脚本有可执行权限，要有可读权限
    在一些没办法调整脚本权限的情况下可以使用
    弊端：要调用者关注当前脚本使用什么解释器(如 python)
# 3 (cd ..;ls -l)
    也是启动一个子进程来执行
# 4 source 脚本地址
    这种方式执行脚本不会产生一个子进程，
    而是将脚本的命令都加载进来，执行
    常用于加载配置文件
    source /etc/profile
    .  跟 source 是同义词

# 6 变量 声明即赋值
varname=value
# 注意等号两边不能留空格，留了空格就变成一个命令+两个参数

# 使用变量
$varname
${varname}
# 使用花括号来限定变量名的范围

# 7 变量的分类
# shell 内变量
    在shell的解析环境中存在的变量
    全局范围的变量（全局变量）
        shell中不使用任何修饰符修饰的变量都是全局变量
        不管是在函数内还是函数外都一样
        从声明语句调用开始一直到脚本结束   都是其生命周期
    局部变量
        用local 修饰
        只能声明在函数内
        从声明语句调用开始一直到函数结束

    shell内变量只能在当前shell进程中使用,跨了进程就不能用

# 环境变量
    是操作系统自带的，每一个进程都会有
        当启动一个子进程的时候，环境变量是从父进程拷贝到子进程
        子进程做任何环境变量的修改，不会影响父进程

        环境变量是单向传递
        export varname=value

# 删除变量
    不管是环境变量还是普通的 shell 内变量，
    都可以使用 unset 变量名   进行删除

# 8 文件名代换
    代换发生在命令执行之前

    * 匹配0个或多个任意字符
    ? 匹配一个任意字符
    [若干字符] 匹配方括号中任意一个字符的一次出现

    代换成功的前提，文件必须存在，如果不存在是代换不了的

# 参数拓展
touch 1.txt 2.txt 3.txt 4.txt
touch {1,2,3,4}.txt
touch {1..4}.txt
touch {1..4}_{5..6}.txt

# 10 命令代换
# 执行某一条命令，将这个命令的标准输出的内容存到某个变量
varname=`cmd arg1 arg2 ...`
$(cmd) 跟 `cmd` 是一样
# 获取当前脚本所在目录

# 11 算数代换  做最基本的整数运算
var=45
$[var+1]
$((var+3))

# 进制转换
echo $[8#10+11]
# 将10以8进制来解析，得到的是10进制的8 ， 加上11 结果为19

# 12 转义
# 将普通字符转成特殊字符
\r \n 
# 将特殊字符转成普通字符
\$SHELL

# 13 引号
    单引号
        保持字符串的字面值
        即使字符串中有特殊符号也会转为普通符号

    双引号
        跟单引号一样，区别在于，支持变量的扩展

        作为一个有经验的shell程序员，使用变量之前，如果变量
        是作为一个参数来传递的，应该要习惯性的加双引号
        防止变量中有空格

        var="a b"
        rm $var      删除两个文件 a b
        rm "$var"    删除一个文件 "a b"

# 14 shell中如何表示真假
    直接使用某条命令的返回状态来表示真假
        main 函数的返回值

    main函数返回 0 表示真
    main函数返回非0 表示假

    跟C完全相反的

    通过 $?  来获取上一条命令的返回状态

# 15 条件测试
    
    test 表达式
    [ 表达式 ]
        中括号是一个单独的命令，后面的参数都是作为该命令的参数
        所以要留空格

        ( EXPRESSION )
            测试该表达式是否为真

       ! EXPRESSION
            取反

       EXPRESSION1 -a EXPRESSION2
            逻辑与

       EXPRESSION1 -o EXPRESSION2
            逻辑或

       -n STRING
            判断字符串不是空串

            注意坑：

                test -n $var

       -z STRING
            判断字符串长度为0

       STRING1 = STRING2
            判断字符串相等

       STRING1 != STRING2
            判断字符串不等

       INTEGER1 -eq INTEGER2
            判断整数相等

       INTEGER1 -ge INTEGER2
            判断整数1>=整数2

       INTEGER1 -gt INTEGER2
            判断整数1>整数2

       INTEGER1 -le INTEGER2
            判断整数1<=整数2

       INTEGER1 -lt INTEGER2
            判断整数1<整数2

       INTEGER1 -ne INTEGER2
            判断整数1!=整数2

       FILE1 -nt FILE2
            判断文件1比文件2新（指最后修改时间）

       FILE1 -ot FILE2
            判断文件1比文件2旧

       -b FILE
            块设备

       -c FILE
            字符设备

       -d FILE
            判断是否目录

       -e FILE
            单纯判断文件是否存在

       -f FILE
            判断文件是一个普通文件

       -h FILE
       -L FILE
            判断是否一个符号链接

       -k FILE
            判断文件的粘着位是否被设置

       -p FILE
            判断文件是否是一个命名管道
       -r FILE
            判断文件是否有读权限

       -s FILE
            判断文件存在并且大小大于0字节

       -S FILE
            判断文件是否是一个socket文件

       -t FD
            判断某个文件描述符被终端打开

       -w FILE
            判断是否有写权限

       -x FILE
            有执行权限

# 16 分支结构 if

    if 命令|条件测试
    then
        xxxxxx
    elif 明令2|条件测试2 ; then    #如果then跟if写在同一行，加分号
        xxxxxx
    else      # else不用加then
        xxxxxx
    fi        # 将if倒着写

    简单的分支结构判断和执行使用  && 和 ||

    make && sudo make install
        如果make执行失败，那么是不会走后面的命令 make install

    echo xxxx ||  exit -1 
        如果前面的命令执行失败，那么就是执行后面的命令

# 17 分支 case
    c语言的switch
    switch(表达式)
    {
    case 值1:
        xxxx
        break;
    case 值2:
        xxxx
        break;
    default;
        xxxx
        break;
    }


    case 表达式 in
    val1|pattern1)
        xxxxxx
        ;;
    val2|pattern2)
        xxxxxx
        ;;
    *)
        xxxx
        ;;
    esac     #将case倒着写

# 18 for
    for varname in 列表 ; do  #do可以单独写一行，如果写在for这一样就要分号
        .....
        echo $varname
    done


    控制循环次数
    for i in {1..100}
    do
        ...
    done


    遍历目录
    for i in `ls`
    do
        ....
    done


# 19 while
    跟C一样

    while 命令|条件测试
    do
        xxxx
    done

    break 和 continue 跟C中的一样

# 20 位置参数以及shift
    
    $0          相当于C语言main函数的argv[0]
    $1、$2...    这些称为位置参数（Positional Parameter），相当于C语言main函数的argv[1]、argv[2]...
    $#          相当于C语言main函数的argc - 1，注意这里的#后面不表示注释
    $@          表示参数列表"$1" "$2" ...，例如可以用在for循环中的in后面。
    $*          表示参数列表"$1" "$2" ...，同上
    $?          上一条命令的Exit Status
    $$          当前进程号

    位置参数默认就支持10个 ，当然$@还是支持n个
    可以配合shift来进行参数左移，来操作不定参数

# 21 输出
    echo -n 表示不换行
    echo -e 解析转义字符
        echo -e "123\t234"

    printf "%d\t%s\n" 123 "hello"
        跟C的printf一样

# ------------------------------------------
1 管道
    使用| 将多个命令拼接在一起
    原理，就是将前一个命令的标准输出作为后一个命令的标准输入来重定向 ,标准错误输出是不会重定向

    需求，编写一个简单的管道命令，读取标准输入，将标准输入的数据转为大写，然后输出

    more 命令
        将标准输入的内容进行缓慢向下查看，要人工操作向下
        只支持向下走，不支持往回走

    less 命令
        比more更加完善，支持往回滚，也支持类似vim的操作，查找 hjkl

2 tee 命令
    将标准输出重新输出，同时存一份到文件
    常用的场景
        开一个服务，服务一直在刷log，需要实时看到log，但是又想将log存成一个文件

3 文件重定向
    cmd > file              把标准输出重定向到新文件中
    cmd >> file             追加

    cmd > file 2>&1         标准出错也重定向到1所指向的file里

        2>&1 
            文件描述符2 也重定向到文件描述符1的位置
            标准错误输出也重定向到标准输出的位置

    cmd >> file 2>&1
    cmd < file
        将file的内容重定向到cmd命令的标准输入
    cmd < file1 > file2     输入输出都定向到文件里
    cmd < &fd               把文件描述符fd作为标准输入

    cmd > &fd               把文件描述符fd作为标准输出

    cmd < &-                关闭标准输入

4 函数
    function 函数名()        #小括号里边也不需要填参数列表
    {
        local var=value   #局部变量

        return 1          #return 只能返回整数，不能返回其他类型 ,返回值是作为退出状态来使用
    }

    function关键字可以省略   ，小括号也可以省略 ，但是两个必须要保留一个，不然解析器不知道是要定义一个函数

    调用函数的方法，就跟普通命令一样

        函数名 arg1 arg2  ...

        函数的执行状态看return语句，如果没有return语句，就以函数里边最后一条执行的指令的返回状态作为整个函数的退出状态

    函数支持递归
        遍历目录，包括子目录，如果是文件就输出xxx是文件，如果是目录就输出xxx是目录

    函数的传参
        也是使用 $1 $2 ... 来获取函数内的参数
    #!/bin/bash

    function visit
    {
        local dir="$1"
        for f in `ls $1`
        do
            if [ -f "$dir/$f" ]
            then
                echo "$dir/$f is a file"
            elif [ -d "$dir/$f" ]
            then
                echo "$dir/$f is a dir"
                visit "$dir/$f"
            else
                echo "$dir/$f is not recognized" 
            fi
        done
    }

    visit .

    脚本的调试
        -n   遍历一下脚本，检查语法错误
        -v   一遍执行脚本一遍将解析到的脚本输出来
        -x   执行脚本的同时打印每一句命令，把变量的值都打印出来  （常用）

        打开调试的方法
            1. bash -x  脚本.sh
            2. 脚本开头 使用 #!/bin/bash -x
            3. 脚本中显式的使用 set -x 打开   使用 set +x 关闭调试

5 正则表达式
练习:               
    1 以S开头的字符串      
        ^S

    2 以数字结尾的字符串
        [0123456789]   匹配任意数字
        [0-9]
        \d
        $               匹配字符串结尾

        [0-9]$

    3 匹配空字符串(没有任何字符)
        ^$
    4 字符串只包含三个数字
        ^\d\d\d$
        ^\d{3}$
            {n} 花括号括起来一个数字，表示前面的单元重复n次

    5 字符串只有3到5个字母
        控制最少重复次数和最大的重复次数

        {m,n} m表示前面单元最小重复次数，n表示最大重复次数

        [a-zA-Z]   表示大小写字母  如果中括号中有多个区间，区间之间不要留空格或其他分隔符

        ^[a-zA-Z]{3,5}$

    6 匹配不是a-z的任意字符
        [^a-z]    中括号中第一个字符如果是^，表示区间取反
        ^[^a-z]$

    7 字符串有0到1个数字或者字母或者下划线
        {0,1} 表示重复0-1次
        ?     也可以表示0-1次重复

        ^[0-9a-zA-Z_]?$
        ^\w?$

    8 字符串有1个或多个空白符号(\t\n\r等)
        \s  表示空白字符 包括 \t\n\r ....
        {1,}   表示重复1-n  跟+号一样

        ^\s+$

    9 字符串有0个或者若干个任意字符(除了\n)
        .  代表任意字符，除了\n
        ^.{,}$   花括号中两个参数置空表示重复次数任意 0-n
        ^.*$     *表示前面的单元重复0-n次

        ? 0-1
        + 1-n
        * 0-n

    10 匹配0或任意多组ABC，比如ABC，ABCABCABC
        使用小括号来讲多个单元重新组合成为一个单元

        ^(ABC)*$

    11 字符串要么是ABC，要么是123
        | 表示选择，选择两边的正则匹配一个

        ^ABC$|^123$
        ^(ABC|123)$     小括号也可以将选择范围控制在括号内

    12 字符串只有一个点号  
        做转义 还是使用\

        ^\.$

    13 匹配十进制3位整数             
        100 - 999
        ^[1-9][0-9]{2}$

        匹配十进制 0-999 的数字
            分段
                一位数
                    [0-9]
                两位数
                    10-99
                    [1-9][0-9]
                三位数
                    [1-9][0-9]{2}

            ^([0-9]|[1-9][0-9]{1,2})$

    14 匹配0-255的整数
        匹配 ip 
            分段
                一位数
                    [0-9]
                两位数
                    10-99
                    [1-9][0-9]
                三位数
                    100-199
                        1[0-9]{2}
                    200-249
                        2[0-4][0-9]
                    250-255
                        25[0-5]

    15 匹配端口号
        0-65535

    16 email
        [\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?

    基础的正则
        +?* 是普通字符
    扩展的正则
        +?* 是特殊字符

    perl的正则
        最常用
        perl正则在扩展正则之上添加了一些特殊符号
            \d \w \s ....

6 sort
    命令从标准输入中读取数据然后按照字符串内容进行排序
    -f 忽略字符大小写
    -n 比较数值大小
    -t 指定分割符，默认是空格或者tab
    -k 指定分割后进行比较字段
    -u 重复的行只显示一次
    -r 反向排序
    -R 打乱顺序
        同样的两行洗不乱

        将/etc/passwd 根据用户id来排序
        sort -t: -k3 -n < /etc/passwd

7 uniq
    去除重复的行,前提是重复的行连续
    -c 显示每行重复的次数
    -d 仅显示重复过的行
    -u 仅显示不曾重复的行
        sort < test.txt | uniq

8 wc
    word count
    -l 统计行数
    -c 统计字节数
    -w 统计单词数

9 grep
    global regular expression print
    -c 只输出匹配行的计数
    -i 不区分大小写
    -H 文件名显示
    -r 递归遍历目录
    -n 显示行号
    -s 不显示不存在或无匹配文本的错误信息
    -v 显示不包含匹配文本的所有行，这个参数经常用于过滤不想显示的行
    -E 使用扩展的正则表达
    -P 使用perl的正则表达式
    -F 匹配固定的字符串，而非正则表达式

    egrep  = grep -E
    fgrep  = grep -F
    rgrep  = grep -r

    grep 默认使用的是基础的正则

10 find
    find pathname -options [-print -exec -ok ...]

    option 选项如下：
        -name 按照文件名查找文件。
            find . -name "1.txt"
        -perm 按照文件权限来查找文件。
            find . -perm 660
        -user 按照文件属主来查找文件。
        -group 按照文件所属的组来查找文件。
        -mtime -n +n 按照文件的更改时间来查找文件，-n表示文件更改时间距现在n天以内，+n表示文件更改时间距现在
            n天以前。find命令还有-atime和-ctime 选项，但它们都和-m time选项。
        -atime 访问时间
        -ctime 创建时间
        -nogroup 查找无有效所属组的文件，即该文件所属的组在/etc/groups中不存在。
        -nouser 查找无有效属主的文件，即该文件的属主在/etc/passwd中不存在。
        -newer file1 ! file2 查找更改时间比文件file1新但比文件file2旧的文件。
        -type 查找某一类型的文件，诸如：
            b - 块设备文件。
            d - 目录。
            c - 字符设备文件。
            p - 管道文件。
            l - 符号链接文件。
            f - 普通文件。
            s - socket文件

        -exec
            find . -name "*.txt" -exec gzip {} \;

            查找当前目录下的txt文件并且打包成为gzip
            每找到一个文件，就会执行exec后面的命令
                gzip ./a/2.txt
                gzip ./a/6.txt

                最后是一个\;  反斜杠不能省,作为当前exec后面命令的结束符

        -ok 
            跟-exec用法一样，但是每找到一个文件要执行后面的命令前会给用户确认

11 xargs
    将标准输入的参数整齐的拼凑在一行里边
    单独使用该命令没什么用，要配合其他命令来使用
    docker ps -aq | xargs docker rm -f

    find . -name "*.txt" | xargs -I{} mv {}  xxx/
        -I{} 指定一个替换字符串作为参数替换
    
12 sed
    文本1 ->  sed + 脚本 -> 文本2 
    ed 编辑器   ->  sed   -> vim 
    sed option 'script' file1 file2 ...             sed 参数  ‘脚本(/pattern/action)’ 待处理文件
    sed option -f scriptfile file1 file2 ...        sed 参数 –f ‘脚本文件’ 待处理文件

        p,  print           打印
        a,  append          追加
        i,  insert          插入
        d,  delete          删除
        s,  substitution    替换

13 awk
    awk option 'script' file1 file2 ...
    awk option -f scriptfile file1 file2 ...

    最常见用法就是过滤哪一列
    xxxx | awk '{print $2}'

    脚本格式
    {actions}                       
        每一行文本都无条件的执行脚本
    /pattern/{actions}
        匹配了模式之后再执行后面的动作
    condition{actions}
        BEGIN
            在遍历文本的第一行之前会执行某个动作
        END
            在遍历完文本之后再去执行某个动作

    ProductA 30
    ProductB 76
    ProductC 55

    在输出表格之前输出表头  产品名字  库存
    输出完表格之后  输出    库存总量 : xxxx

    在遍历之前输出表头
    BEGIN{
        printf "%s\t%s\n","产品","库存";
        sum=0;
    }
    在遍历每一行的过程中输出每一行的内容，将库存加到sum变量
    {
        printf "%s\t%s\n",$1,$2;
        sum+=$2;
    }
    遍历完之后输出sum变量
    END{
        printf "库存总量:%d\n",sum
    }

14 crontab
    linux 系统定时器
    需求，每天什么时候去做什么事情
    /etc/crontab

# m h dom mon dow user  command                                                                            
17 *    * * *   root    cd / && run-parts --report /etc/cron.hourly                                        
25 6    * * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )        
47 6    * * 7   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.weekly )       
52 6    1 * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.monthly )      

```