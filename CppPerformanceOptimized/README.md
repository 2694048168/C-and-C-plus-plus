## 现代 C++ 性能优化指南

> the **Modern C++** Best Practices and the Proven Techniques **Optimized for Heightened Performance**.

```shell
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Debug
cmake --build build --config Debug

cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build build --config Release

# 自动以最大线程数进行并行编译
sudo cmake --build build --target all -- -j $(nproc)
```

### **Reference**
- [Kurt C++ tech-blog](http://oldhandsblog.blogspot.com/)
- [GCC compiler](https://gcc.gnu.org/)
- [GCC compile options](https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html)
- [Clang compiler](https://clang.llvm.org/)
- [Clang compile options](https://clang.llvm.org/docs/UsersManual.html)
- [Clang compile options](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Inter C++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [Inter C++ compile options](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-1/use-compiler-options.html)
- [MSVC C++ compiler](https://visualstudio.microsoft.com/zh-hans/vs/features/cplusplus/)
- [MSVC C++ compile options](https://learn.microsoft.com/zh-cn/cpp/build/reference/compiler-options?view=msvc-170)


### **0. Attention**
- 不同优化方式的正确使用时机则取决于编译器(compiler)和标准库(std-lib)的实现以及代码是在哪种处理器(CPU)上测试的
- 性能优化(Optimized Performance)是一门实验科学, 盲目地信任优化建议往往会失望
- C++ 代码特有的行为模式对 C++ 程序进行优化,有些对 C++ 代码特别有效的优化技术,可能对其他编程语言没有效果
- C++ 设计最佳实践的正确代码进行优化,不仅可以体现出优秀的 C++ 设计, 更快速、更低耗地运行
- 开发人员缺乏现代微处理器设备的基本常识, 或是没有仔细考虑各种 C++ 对象构建方式的性能开销
- C++ 提供了对内存管理和复制的精准控制功能
- 任何不更新的技术或者功能都注定失败, 因为新的算法被不断地发明出来, 新的编程语言特性也在不断地涌现
- 如何逐步地改善代码, 熟悉代码的优化过程并形成一种可以提高优化效果的思维模式
- 如何优化编码过程, 注重程序在运行时的性能开销的开发人员, 可以从一开始就编写出高效代码
- 实际需求开发中, 当一个带有性能指标的特性编码完成后或是需要实现特定的性能目标时, 就会需要进行优化
- 性能优化的目的是通过改善正确程序的行为使其满足客户对处理速度、吞吐量、内存占用以及能耗等各种指标的需求
- bug 修复与性能(pref)优化之间重要区别, 性能是一个连续变量, 特性(feat)要么是实现了,要么是没有实现;bug 要么存在, 要么不存在
- 优化是一门实验科学,需要有更深入的科学思维方式; 要想优化成功,需要先观察程序行为,然后基于这些程序行为作出可测试的推测,再进行实验得出测试结果; 实验, 而非直觉, 才是贯穿优化的主题
- 选择一种更好的算法或是数据结构, 将程序的某个特性的性能从慢到无法忍受提升至可发布的状态
- 现代处理器(CPU)处理速度能够每纳秒分发一条指令,处理器的处理速度越快,被浪费的指令的累积速度也会越快; 如果一个程序所执行的指令中 50% 都是不必要的,那么不管这些不必要的指令的执行速度有多快,只要删除这些指令,程序的处理速度就会变为原来的两倍
- C++ 的混合特性提供了多种实现方式,一方面可以实现性能管理的全自动化;另一方面也可以对性能进行更加精准的控制
- C++ 有一些热点代码是性能"惯犯", 其中包括函数调用、内存分配和循环


### **1. Features**
- [x] 用好的编译器(compiler)并用好编译器(Command Options)
    - Compiler: GCC, MSVC, Clang, Intel C++ Complier, MinGW
    - Compile Command Options, such as -O2
    - C++ 编译器的文档对可用优化选项和预处理指令做了全面说明
- [x] 使用更好的算法
    - 最优算法适用于大大小小的各种程序,从精简的闭式计算到短小的关键字查找函数,再到复杂的数据结构和大规模的程序
    - 程序性能的重要技巧: 
        - 预计算 pre-computation, 将计算从运行时移动至链接、编译或是设计时
        - 延迟计算 lazy computation, 如果通常计算结果不会被使用,那么将计算推迟至真正需要使用计算结果时
        - 缓存 caching, 节省和复用昂贵的计算
- [x] 使用更好的库
    - C++ 编译器提供的标准 C++ 模板库和运行时库必须是可维护的、全面的和非常健壮的
    - 掌握标准 C++ 模板库(STL):
        - 查找和排序算法
        - 容器类的最优惯用法
        - I/O 流
        - 并发
        - 内存管理
    - 可供选择的开源库,实现可能比供应商提供的 C++ 运行时库更快、更强:
        - [Boost Project](https://www.boost.org/)
        - [Google Code](https://github.com/google)
        - [Microsoft Code](https://github.com/microsoft)
        - [Meta or Facebook Code](https://github.com/facebook)
        - [Alibaba Code](https://github.com/ALIBABA)
        - [Bytedance Code](https://github.com/bytedance)
        - [Tencent Code](https://github.com/tencent)
        - [Baidu Code](https://github.com/BAIDU)
    - 优秀的函数库的 API 所提供的函数反映了这些 API 的惯用法,使得用户可以无需频繁地调用其中的基础函数
        - 避免频繁发生昂贵的函数调用开销
        - 高度优化后的程序的复杂性,函数和类库是非常合适的地方,库函数通常位于深层嵌套调用链的底端
- [x] 减少内存分配和复制
    - 减少对内存管理器的调用是一种非常有效的优化手段
    - 绝大多数 C++ 语言特性的性能开销最多只是几个指令,但是每次调用内存管理器的开销却是数千个指令
    - 对缓存复制函数的一次调用也可能消耗数千个 CPU 周期,减少复制是一种提高代码运行速度的优化方式
    - 可能会发生复制的热点代码是构造函数和赋值运算符以及输入输出
- [x] 移除计算: 除了内存分配和函数调用外, 单条 C++ 语句的性能开销通常都很小
    - 热点代码,即被频繁地执行的代码
    - 在尝试减少计算数量之前,如何确定程序中的哪部分会被频繁地执行
    - 现代 C++ 编译器(Modern Compiler)在进行这些局部改善方面也做得非常优秀
- [x] 使用更好的数据结构
    - 插入、迭代、排序和检索元素的算法的运行时开销取决于数据结构
    - 不同的数据结构在使用内存管理器的方式上也有所不同
    - 数据结构可能有也可能没有优秀的缓存本地化
- [x] 提高并发性 & 优化内存管理 & **恶魔隐藏在细节中**


### **2. 影响优化的计算机行为**

> 撒谎, 即讲述美丽而不真实的故事, 乃是艺术的真正目的.

- 性能优化技术相关的计算机硬件, 处理器的体系结构, 从中获得性能优化的启发
- 微处理器 ---> 逻辑门电路 ---> 时钟频率 ---> 内存地址(位bit/字word/字节byte) ---> 寄存器(register)
- 从内存地址读取数据和向内存地址写入数据是需要花费时间, 指令对数据进行操作也是需要花费时间的
- 现代处理器(CPU架构,操作系统调度)做了许多不同的、交互的事情来提高指令执行速度, 导致指令的执行时间实际上变得难以确定
- 与其他任何有效的内存地址都不同的特殊的地址 nullptr; 整数 0 会被转换为 nullptr, 尽管在地址 0 上不需要 nullptr
- 声明 volatile 变量会要求编译器在每次使用该变量时都获取它的一份新的副本,而不用通过将该变量的值保存在一个寄存器中并复用它来优化程序
- 这些地址的值可能会在同一个线程对该地址的两次连续读的间隔发生变化, 这表示硬件发生了变化
- std::atomic<> 的特性,可以让内存在一段短暂的时间内表现得仿佛是字节的简单线性存储一样,以远离多线程执行、多层高速缓存等引发的问题
- 操作系统会使用计算机硬件来隐藏这些谎言,C++ 知道计算机远比这个简单模型(C++ 模型)要复杂
- 除了降低程序的运行速度外,这些谎言其实对程序运行并没有什么影响;不过它们会导致性能测量变得复杂
- 计算机的主内存相对于它内部的逻辑门和寄存器来说非常慢,桌面级处理器在从主内存中读取一个数据字的时间内,可以执行数百条指令
- 通往主内存的接口是限制执行速度的瓶颈(冯•诺伊曼瓶颈),内存总线的访问带宽是有限的,对性能的隐式限制被称为内存墙(memory wall)
- 内存访问并非以字节为单位(LEVEL1_ICACHE_LINESIZE=64),这种访问被称为非对齐的内存访问(unaligned memory access)
- command line: **getconf -a** to check the CPU level-cache
- 在定义结构体时, 对各个数据字段的大小和顺序稍加注意, 可以在保持对齐的前提下使结构体更加紧凑
- 补偿主内存的缓慢速度,许多计算机中都有高速缓存(cache memory),来加快对那些使用最频繁的内存字的访问速度

> 在桌面级处理器中, 通过一级高速缓存(LEVEL1_ICACHE_SIZE and LEVEL1_DCACHE_SIZE | LEVEL1_ICACHE_ASSOC | LEVEL1_ICACHE_LINESIZE)、二级高速缓存(LEVEL2_CACHE_SIZE | LEVEL2_CACHE_ASSOC | LEVEL2_CACHE_LINESIZE)、三级高速缓存(LEVEL3_CACHE_SIZE | LEVEL3_CACHE_ASSOC | LEVEL3_CACHE_LINESIZE)、主内存(memory)和磁盘上的虚拟内存页(virtual memory page)访问内存的时间开销范围可以跨越五个数量级. 这就是专注于指令的时钟周期和其他"奥秘"经常会令人恼怒而且没有效果的一个原因, 高速缓存的状态会让指令的执行时间变得非常难以确定. 当执行单元需要获取不在高速缓存(cache memory)中的数据(其中的数据状态会有M-E-S-I)时, 有一些当前处于高速缓存中的数据必须被舍弃以换取足够的空余空间; 通常选择放弃的数据都是最近很少被使用的数据, 这一点与性能优化有着紧密的关系, 因为这意味着访问那些被频繁地访问过的存储位置的速度会比访问不那么频繁地被访问的存储位置更快. 读取一个不在高速缓存中的字节甚至会导致许多临近的字节也都被缓存起来(这也意味着, 许多当前被缓存的字节将会被舍弃); 这些临近的字节也就可以被高速访问了. 对于性能优化而言, 这一点非常重要, 因为这意味着平均而言, 访问内存中相邻位置的字节要比访问互相远隔的字节的速度更快; 就 C++ 而言, 这表示一个包含循环处理的代码块的执行速度可能会更快, 这是因为组成循环处理的指令会被频繁地执行, 而且互相紧挨着, 因此更容易留在高速缓存中; 一段包含函数调用或是含有 if 语句导致执行发生跳转的代码则会执行得较慢, 因为代码中各个独立的部分不会那么频繁地被执行, 也不是那么紧邻着; 相比紧凑的循环, 这样的代码在高速缓存中会占用更多的空间; 如果程序很大, 而且缓存有限, 那么一些代码必须从高速缓存中舍弃以为其他代码腾出空间, 当下一次需要这段代码时, 访问速度会变慢; 类似地, 访问包含连续地址的数据结构(如数组或矢量), 要比访问包含通过指针链接的节点的数据结构快, 因为连续地址的数据所需的存储空间更少. 访问包含通过指针链接的记录的数据结构(例如链表或者树)可能会较慢, 这是因为需要从主内存读取每个节点的数据到新的缓存行中.

- 内存字分为大端和小端(字节序, endian-ness): 从首字节地址读取最高有效位的计算机被称为大端计算机; 小端计算机则会首先读取最低有效位
- 内存容量是有限的: 将没有放入物理内存中的数据作为文件存储在磁盘上, 这种机制被称为虚拟内存(virtual memory)
- 从磁盘上获取一个内存块需要花费数十毫秒, 对现代计算机来说, 这几乎是一个恒定值

> 高速缓存(cache memory)和虚拟内存(virtual memory)带来的一个影响是, 由于高速缓存的存在, 在进行性能测试时, 一个函数运行于整个程序的上下文中时的执行速度可能是运行于测试套件中时的万分之一; 当运行于整个程序的上下文中时, 函数和它的数据不太可能存储至缓存中, 而在测试套件的上下文中, 它们则通常会被缓存起来. 这个影响放大了减少内存或磁盘使用量带来的优化收益, 而减小代码体积的优化收益则没有任何变化. 第二个影响则是, 如果一个大程序访问许多离散的内存地址, 那么可能没有足够的高速缓存来保存程序刚刚使用的数据; 这会导致一种性能衰退, 称为页抖动(page thrashing), 当在微处理器内部的高速缓存中发生页抖动时, 性能会降低; 当在操作系统的虚拟缓存文件中发生页抖动时, 性能会下降为原来的 1/1000; 过去, 计算机的物理内存很少, 页抖动更加普遍; 不过如今, 这个问题仍然会发生.

- 指令执行缓慢: 执行指令的速度可以比从主内存获取指令快很多倍,多数时候都需要高速缓存去"喂饱"执行单元, 对优化而言, 这意味着内存访问决定了计算开销
- 指令在流水线中被解码、获取参数、执行计算,最后保存处理结果; 处理器的性能越强大,这条流水线就越复杂; 它会将指令分解为若干阶段,这样就可以并发地处理更多的指令

> 如果指令 B 需要指令 A 的计算结果, 那么在计算出指令 A 的处理结果前是无法执行指令 B 的计算的; 这会导致在指令执行过程中发生**流水线停滞(pipeline stall)** —— 一个短暂的暂停, 因为两条指令无法完全同时执行; 如果指令 A 需要从内存中获取值, 然后进行运算得到线程 B 所需的值, 那么**流水线停滞时间**会特别长; 流水线停滞会拖累高性能微处理器, 让它变得与烤面包机中的处理器的速度一样慢. 另一个会导致流水线停滞的原因是计算机需要作决定, 大多数情况下,在执行完一条指令后, 处理器都会获取下一个内存地址中的指令继续执行; 这时多数情况下, 下一条指令已经被保存在高速缓存中了; 一旦流水线的第一道工序变为可用状态, 指令就可以连续地进入到流水线中; 但是控制转义指令略有不同, 跳转指令或跳转子例程指令会将执行地址变为一个新的值; 在执行跳转指令一段时间后, 执行地址才会被更新, 在这之前是无法从内存中读取"下一条"指令并将其放入到流水线中的. 新的执行地址中的内存字不太可能会存储在高速缓存中, 在更新执行地址和加载新的"下一条"指令到流水线中的过程中, 会发生流水线停滞; 在执行了一个条件分支指令后, 执行可能会走向两个方向: 下一条指令或者分支目标地址中的指令; 最终会走向哪个方向取决于之前的某些计算的结果, 这时流水线会发生停滞, 直至与这些计算结果相关的全部指令都执行完毕, 而且还会继续停滞一段时间, 直至决定一下条指令的地址并取得下一条指令为止; 对性能优化而言, 这一项的意义在于**计算比做决定更快**.

- 程序执行中的多个流(进程[Process]、线程[Thread]、协程[Coroutine])

> 任何运行于现代操作系统中的程序都会与同时运行的其他程序、检查磁盘的定期维护进程以及控制网络接口、磁盘、声音设备、加速器、温度计和其他外设的操作系统的各个部分共享计算机. 每个程序都会与其他程序竞争计算机资源. 当许多程序一齐开始运行, 互相竞争内存和磁盘时, 为了性能调优, 如果一个程序必须在启动时执行或是在负载高峰期时执行, 那么在**测量性能时也必须带上负载**. 通过任务列表就可以发现, 微处理器所执行的软件进程远比这个物理核心数量大, 而且绝大多数进程都有多个线程在执行, 操作系统会执行一个线程一段很短的时间, 然后将**上下文切换**至其他线程或进程,对程序而言,就仿佛执行一条语句花费了一纳秒(ns),但执行下一条语句花费了 60 毫秒(ms). 切换上下文究竟是什么意思呢？如果操作系统正在将一个线程切换至同一个程序的另外一个线程, 这表示要为即将**暂停的线程保存**处理器中的寄存器,然后为即将被继续执行的线程**加载**之前保存过的寄存器, 现代处理器中的寄存器包含数百字节的数据, 当**新线程继续执行**时,它的数据可能并不在**高速缓存**中,所以当加载新的上下文到高速缓存中时,会有一个**缓慢的初始化阶段**, 因此切换线程上下文的成本很高. 当操作系统从一个程序切换至另外一个程序时, 这个过程的开销会更加昂贵; 所有脏的高速缓存页面(页面被入了数据,但还没有反映到主内存中)都必须被刷新至物理内存中; 所有的处理器寄存器都需要被保存; 然后, 内存管理器中的"物理地址到虚拟地址"的内存页寄存器也需要被保存; 接着, 新线程的"物理地址到虚拟地址"的内存页寄存器和处理器寄存器被载入; 最后就可以继续执行了, 但是这时高速缓存是空的, 因此在高速缓存被填充满之前,还有一段缓慢且需要**激烈地竞争内存**的初始化阶段. 当一个程序必须等某个事件发生时, 它甚至可能会在这个事件发生后继续等待,直至操作系统让处理器为继续执行程序做好准备; 这会导致当程序运行于其他程序的上下文中, 竞争计算机资源时, 程序的运行时间变得更长和更加难以确定. 为了能够达到更好的性能, 一个多核处理器的执行单元及相关的高速缓存, 与其他的执行单元及相关的高速缓存都是或多或少互相独立的; 不过,所有的执行单元都共享同样的主内存, 执行单元必须竞争使用那些将可以它们链接至主内存的硬件, 使得在拥有多个执行单元的计算机中, 冯•诺依曼瓶颈的限制变得更加明显. 当执行单元写值时, 这个值会首先进入高速缓存内存; 不过最终,这个值将被写入至主内存中,这样其他所有的执行单元就都可以**看见**这个值了; 但是,这些执行单元在访问主内存时存在着**竞争**,所以可能在执行单元改变了一个值,然后又执行几百个指令后,主内存中的值才会被更新(多线程安全问题). 因此,如果一台计算机有多个执行单元,那么一个执行单元可能需要在很长一段时间后才能看见另一个执行单元所写的数据被反映至主内存中,而且主内存发生改变的顺序可能与指令的执行顺序不一样; 受到不可预测的时间因素的干扰,执行单元看到的共享内存字中的值可能是旧的,也可能是被更新后的值,这时,必须使用**特殊的同步指令**来确保运行于不同执行单元间的线程看到的内存中的值是一致的(**内存一致性**). 对优化而言,这意味着**访问线程间的共享数据比访问非共享数据要慢得多**.

- 调用操作系统的开销是昂贵的(内核态和用户态切换, 涉及频繁的系统调用)

> 除了最小的处理器(嵌入式芯片)外, 其他处理器都有硬件可以确保程序之间是互相隔离的. 这样程序 A 不能读写和执行属于程序 B 的物理内存; 这个硬件还会保护**操作系统内核**不会被程序覆写; 另一方面**操作系统内核**需要能够访问所有程序的内存, 这样程序就可以通过**系统调用**访问操作系统; 有些操作系统还允许程序发送访问共享内存的请求, 许多系统调用的发生方式和共享内存的分布方式是多样和神秘的. 对优化而言, 这意味着**系统调用的开销是昂贵**的, 是单线程程序中的函数调用开销的数百倍.

- C++ 对用户所撒的最大的谎言就是运行它的计算机的结构是简单的、稳定的. 为了假装相信这条谎言, C++ 让开发人员不用了解每种微处理器设备的细节即可编程,如同正在使用真实得近乎残酷的汇编语言编程一样
- 并非所有语句的性能开销都相同

> 一个赋值语句, 如 BigInstance i = OtherObject; 会复制整个对象的结构; 更值得注意的是, 这类赋值语句会调用 BigInstance 的构造函数, 而其中可能隐藏了不确定的复杂性. 当一个表达式被传递给一个函数的形参时, 也会调用构造函数; 当函数返回值时也是一样的; 而且由于算数操作符和比较操作符也可以被重载, 所以 A=B*C; 可能是 n 维矩阵相乘，if (x<y) 可能比较的是具有任意复杂度的有向图中的两条路径. 对优化而言, 这一点的意义是**某些语句隐藏了大量的计算**, 但从这些语句的外表上看不出它的性能开销会有多大.

- 语句并非按顺序执行

> C++ 程序表现得仿佛它们是按顺序执行的,完全遵守了 C++ 流程控制语句的控制; 其中含糊其辞的"仿佛"正是许多编译器(Compiler)进行优化的基础, 也是现代计算机硬件的许多技巧的基础. 当然在底层, **编译器能够而且有时也确实会对语句进行重新排序以改善性能**; 但是编译器知道在测试一个变量或是将其赋值给另外一个变量之前,必须先确定它包含了所有的最新计算结果; 现代处理器也可能会选择**乱序执行指令**, 不过它们包含了可以确保在随后读取同一个内存地址之前, 一定会先向该地址写入值的逻辑(单线程不变原则). 甚至微处理器的内存控制逻辑可能会选择延迟写入内存以优化内存总线的使用; 但是内存控制器知道哪次写值正在从执行单元穿越高速缓存飞往主内存的"航班"中, 而且确保如果随后读取同一个地址时会使用这个"航班"中的值. 并发会让情况变得复杂, **C++ 程序在编译时不知道是否会有其他线程并发运行**; C++ 编译器不知道哪个变量——如果有的话——会在线程间共享,当程序中包含共享数据的并发线程时, 编译器对语句的重排序和延迟写入主内存会导致计算结果与按顺序执行语句的计算结果不同(多线程安全问题); 开发人员必须向多线程程序中**显式地加入同步代码**来确保可预测的行为的一致性; 当并发线程共享数据时, 同步代码**降低**了并发量.

- 小结 summary:
    - 在处理器中, 访问内存的性能开销远比其他操作的性能开销大
    - 非对齐访问所需的时间是所有字节都在同一个字中时的两倍
    - 访问频繁使用的内存地址的速度比访问非频繁使用的内存地址的速度快
    - 访问相邻地址的内存的速度比访问互相远隔的地址的内存快
    - 由于高速缓存的存在, 一个函数运行于整个程序的上下文中时的执行速度可能比运行于测试套件中时更慢
    - 访问线程间共享的数据比访问非共享的数据要慢很多
    - 计算比做决定快
    - 每个程序都会与其他程序竞争计算机资源
    - 如果一个程序必须在启动时执行或是在负载高峰期时执行, 那么在测量性能时必须加载负载
    - 每一次赋值、函数参数的初始化和函数返回值都会调用一次构造函数, 这个函数可能隐藏了大量的未知代码
    - 有些语句隐藏了大量的计算, 从语句的外表上看不出语句的性能开销会有多大
    - 当并发线程共享数据时, 同步代码降低了并发量
    - 缓存局部性(cache locality)更好

### **3. 测量性能**

> 测量可测量之物, 将不可测量之物变为可测量.

- 测量和实验是所有改善程序性能尝试的基础,设计性能测量实验,使得测量结果更有指导意义
- 测量性能的工具软件: 分析器(perf)和计时器软件(timer)
- 编译器厂商通常在编译器中都会提供的分析器(profiler)
    - 分析器会生成各个函数在程序运行过程中被调用的累积时间的表格报表
    - 对性能优化而言,是一个非常关键的工具,列出程序中最热点的函数
- 计时器软件(software timer), 开发人员可以自己实现这个工具
- 古老的"实验笔记本", 文本文件记录实验数据仍然是不可或缺的优化工具

> 人的感觉对于检测性能提高了多少来说是不够精确的, 人的记忆力不足以准确地回忆起以往多次实验的结果. 可能会误导使你相信了一些并非总是正确的事情. 当判断是否应当对某段代码进行优化的时候, 开发人员的直觉往往差得令人吃惊; 他们编写了函数, 也知道这个函数会被调用, 但并不清楚调用频率以及会被什么代码所调用; 于是一段低效的代码混入了核心组件中并被调用了无数次; 经验也可能会欺骗你, 编程语言、编译器、库和处理器都在不断地发展, 之前曾经肯定是热点的函数可能会变得非常高效, 反之亦然. 只有**测量**才能告诉你到底是在**优化**游戏中取胜了还是失败了. 那些具有优化技巧的开发人员都会系统地完成优化任务: 1)做出的预测都是可测试的, 而且会记录下预测; 2)保留代码变更记录; 3)使用可以使用的最优秀的工具进行测量; 4)保留实验结果的详细笔记. 停下来思考, 优化, 这是一项必须**不断实践的技能**.

- 如果只能让程序的运行速度提高 1% 是不值得冒险去修改代码的, 因为修改代码可能会引入 bug; 只有能显著地提升性能时才值得修改代码
- 性能优化的基本规则是 90/10 规则: 一个程序花费 90% 的时间执行其中 10% 的代码
- 90/10 规则表示某些代码块是会被频繁地执行的热点(hot spot), 而其他代码则几乎不会被执行;这些热点就是要进行性能优化的对象

> 90/10 规则的一个结论是, 优化程序中的所有例程并没有太大帮助; 优化一小部分代码事实上已经足够提供所想要的性能提升了; 识别出 10% 的热点代码是值得花费时间的, 但靠猜想选择优化哪些代码可能只是浪费时间.

- 阿姆达尔定律: 阿姆达尔定律是由计算机工程先锋基恩•阿姆达尔(Gene Amdahl)提出并用他的名字命名的,它定义了优化一部分代码对整体性能有多大改善
- 阿姆达尔定律证明了 90/10 规则, 而且展示了对 10% 的热点代码进行适当的优化, 就可以带来如此大的性能提升

> 阿姆达尔定律告诉我们, 如果被优化的代码在程序整体运行时间中所占的比率不大, 那么即使对它的优化非常成功也是不值得的; 阿姆达尔定律的教训是, 当你的同事兴冲冲地在会议上说他知道如何将一段计算处理的速度提高 10 倍, 这并不一定意味着性能优化工作就此结束了. 在开始性能调优前, 必须要有正确的代码, 即在某种意义上可以完成我们所期待的处理的代码; 你需要擦亮眼睛审视这些代码, 然后问自己: "为什么这些代码是热点?"; 对于"为什么这些代码是热点"这个问题的回答构成了你要测试的假设; 实验要对程序的两种运行时间进行测量: 一种是修改前的运行时间, 一种是修改后的运行时间; 如果后者比前者短, 那么实验验证了你的假设.

- 每次的测试运行情况都被记录在案, 那么就可以快速地重复实验; 关心实验的可重复性, 避免偶然性和bug
- 使用纸和笔记录是一种很稳健、容易使用而且有着千年历史的技术; 即使在开发团队替换了版本管理工具或测试套件的情况

> 优化工作受两个数字主导: 优化前的**性能基准测量值**和**性能目标值**, 测量性能基准不仅对于衡量每次独立的改善是否成功非常重要,而且对于向其他利益相关人员(团队合作者,项目经理等)就优化成本开销做出解释也是非常重要的. 优化目标值之所以重要, 是因为在优化过程中优化效果会逐渐变小; 前期提升效果很快很明显, 但是优化到最后每一小步都很简单, 需要付出更多的努力, 毕竟树上总是有些容易摘取的挂得很低的水果. 

- 用户体验(UX)设计的一个学科分支专门研究用户如何看待等待时间
    - 启动时间
    - 退出时间
    - 响应时间: 执行一个命令的平均时间或最长时间
        - 低于 0.1 秒: 用户在直接控制
        - 0.1 秒至 1 秒: 用户在控制命令
        - 1 秒至 10 秒: 计算机在控制; 10 秒是用户能保持注意力的最长时间, 如果多次遇到这种长时间等待 UI 发生改变的情况, 用户满意度会急速下降
        - 高于 10 秒: 喝杯咖啡休息一下; 甚至会关闭程序, 然后去其他地方找找满足感, 产品失去竞争力
    - 吞吐量: 吞吐量表述为在一定的测试负载下, 系统在每个时间单位内所执行的操作的平均数

> 有时, 也可能会发生过度优化的情况; 例如在许多情况下, 用户认为响应时间小于 0.1秒就是一瞬间的事了; 在这种情况下, 即使将响应时间从 0.1 秒改善为了 1 毫秒, 也不会增加任何价值, 尽管响应速度提升了 100 倍. 优化一个函数、子系统、任务或是测试用例永远不等同于改善整个程序的性能,由于测试时的设置在许多方面都与处理客户数据的正式产品不同, 在所有环境中都取得在测试过程中测量到的性能改善结果是几乎不可能的; 尽管某个任务在程序中负责大部分的逻辑处理, 但是使其变得更快可能仍然无法使整个程序变得更快.

- 分析器是一个可以生成另外一个程序的执行时间的统计结果的程序; 分析器可以输出一份包含每个语句或函数的执行频度、每个函数的累积执行时间的报表
- Windows 上的 Visual Studio 和 Linux 上的 GCC 都带有分析器
- 分析器的分析功能都是量身设计的, 它自身的性能开销非常小, 因此它对程序整体运行时间的影响也很小

> 分析器的最大优点是它直接显示出了代码中最热点的函数; 分析经验来看, 对调试构建(debug build)的分析结果和对正式构建(release build)的分析结果是一样的. 在某种意义上, 调试构建更易于分析, 因为其中包含所有的函数, 包括内联函数, 而正式构建则会隐藏这些被频繁调用的内联函数(编译器优化功能).

> 在 Windows 上分析调试构建的一个问题是, 调试构建所链接的是调试版本的运行时库. 调试版本的内存管理器函数会执行一些额外的测试, 以便更好地报告重复释放的内存和内存泄漏问题. 这些额外测试的开销会显著地增加某些函数的性能开销. 有一个环境变量可以让调试器不要使用调试内存管理器: 进入控制面板→系统属性→高级系统设置→环境变量→系统变量, 然后添加一个叫作 _NO_DEBUG_HEAP 的新变量并设定其值为 1.

> 当遇到 **IO 密集型**程序或是**多线程**程序时, 分析器的结果中可能会含有误导信息, 因为分析器减去了**系统调用**的时间和等待事件的时间

> 真正的测量实验必须能够应对可变性(variation): 可能破坏完美测量的误差源. 可变性有两种类型: 随机的和系统的. 随机的可变性对每次测量的影响都不同, 就像一阵风导致弓箭偏离飞行线路一样. 系统的可变性对每次测量的影响是相似的, 就像一位弓箭手的姿势会影响他每一次射箭都偏向靶子的左边一样. 可变性自身也是可以测量的. 衡量一次测量过程中的可变性的属性被称为精确性(precision)和正确性(trueness). 这两种属性组合成的直观特性称为准确性(accuracy).

- 精确性、正确性和准确性; 测量时间、测量持续时间; 测量分辨率(指测量所呈现出的单位的大小)

> **非确定性行为**,计算机是带有大量内部状态的异常复杂的装置,其中绝大多数状态对开发人员是不可见的. 执行函数会改变计算机的状态(例如高速缓存中的内容), 这样每次重复执行指令时, 情况都会与前一条指令不同. 因此内部状态的不可控的变化是测量中的一个随机变化源. 而且操作系统对任务的安排也是不可预测的, 这样在测量过程中, 在处理器和内存总线上运行的其他活动会发生变化. 这会降低测量的准确性. 操作系统甚至可能会暂停执行正在被测量的代码, 将 CPU 时间分配给其他程序. 但是在暂停过程中, 时标计数器仍然在计时. 这会导致与操作系统没有将 CPU 时间分配给其他程序相比, 测量出的执行时间变大了. 这是一种会对测量造成更大影响的随机变化源.

- 优化后代码的运行时间与优化前代码的运行时间的比率被称为相对性能
- 统计数字是从大量的独立事件中得到的, 因此这些数字的持续改善表明对代码的修改是成功的
- 通过取多次迭代的平均值来提高准确性
- 通过提高优先级减少操作系统的非确定性行为

> 通过提高测量进程的优先级, 可以减小操作系统使用 CPU 时间片段去执行测量程序以外的处理的几率. 在 Windows 上, 可以通过调用 SetPriorityClass() 函数来设置进程的优先级, 而 SetThreadPriority() 函数则可以用来设置线程的优先级. 

```C++
// 提高了当前进程和线程的优先级:
SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
// 在测量结束后，通常应当将进程和线程恢复至正常优先级:
SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
```

- 如果在测量时间时调用了某个函数 10 000 次, 这段代码和相关的数据会被存储在高速缓存中; 单独测试一个功能函数并不能完全验证在整体系统中的性能需求.
- 评估独立的C++语句的开销: **访问内存的时间开销**远比执行其他指令的开销大
- 评估一条 C++ 语句的开销有多大, 那就是计算该语句对**内存的读写次数**
- 单位时间内的开销也取决于 C++ 语句要访问的内容是否在高速缓存(cache memory)中
- 评估循环的开销:
    - 评估嵌套循环中的循环次数: 循环次数(内层循环的次数 * 外层循环的次数) * 语句的开销
    - 评估循环次数为变量(循环处理会不断重复直至满足某个条件)的循环的开销
    - 识别出隐式循环(最频繁地被分发的事件中的代码都可能是热点代码)
    - 识别假循环

### **4. 优化字符串的使用: 案例研究**
- std::string 对内存管理器的调用次数占到了内存管理器被调用的总次数的一半
- 只要操作字符串的代码会被频繁地执行, 那么那里就有优化的用武之地
- 字符串的某些行为会增加使用它们的开销, 这一点与实现方式无关
- 字符串是动态分配的, 它们在表达式中的行为与值相似, 而且实现它们需要大量的复制操作

> 字符串内部的字符缓冲区的大小仍然是固定的, 任何会使字符串变长的操作, 如在字符串后面再添加一个字符或是字符串, 都可能会使字符串的长度超出它内部的缓冲区的大小. 当发生这种情况时, 操作会从**内存管理器**中获取一块新的缓冲区, 并将字符串**复制**到新的缓冲区中. 为了能让字符串增长时重新分配内存的开销"分期付款", std::string 使用了一个小技巧, 字符串向内存管理器申请的字符缓冲区的大小(**capacity**)并非与字符串所需存储的字符数(**size**)完全一致, 而是比该数值更大. 例如,有些字符串的实现方式所申请的字符缓冲区的大小是需要存储的字符数的两倍; 这样,在下一次申请新的字符缓冲区之前,字符串的容量足够允许它增长一倍; 下一次某个操作需要增长字符串时, 现有的缓冲区足够存储新的内容,可以**避免申请新的缓冲区**; 这个小技巧带来的好处是随着字符串变得更长, 在字符串后面再添加字符或是字符串的开销近似于一个常量; 而其代价则是字符串携带了一些未使用的内存空间. 如果字符串的实现策略是字符串缓冲区增大为原来的两倍, 那么在该字符串的存储空间中,有一半都是未使用的.

- 字符串就是值: 在赋值语句和表达式中，字符串的行为与值是一样的

> 将一个字符串赋值给另一个字符串的工作方式是一样的(值的行为方式),就仿佛每个字符串变量都拥有一份它们所保存的内容的私有副本一样. 由于字符串就是值,因此字符串表达式的结果也是值. 如果使用 s1 = s2 + s3 + s4; 这条语句连接字符串, 那么 s2 + s3 的结果会被保存在一个新分配的临时字符串中; 连接 s4 后的结果则会被保存在另一个临时字符串中. 这个值将会取代 s1 之前的值; 接着, 为第一个临时字符串和 s1 之前的值动态分配的内存将会被释放. 这会导致多次调用内存管理器.

- 字符串会进行大量复制

> 由于字符串的行为与值相似, 因此修改一个字符串不能改变其他字符串的值. 但是字符串也有可以改变其内容的变值操作. 正是因为这些变值操作的存在, 每个字符串变量必须表现得好像它们拥有一份自己的私有副本一样. 实现这种行为的最简单的方式是当创建字符串、赋值或是将其作为参数传递给函数的时候进行一次复制. 如果字符串是以这种方式实现的, 那么赋值和参数传递的开销将会变得很大, 但是变值函数(mutating function)和非常量引用的开销却很小. 有一种被称为"写时复制"(copy on write)的著名的编程惯用法, 它可以让对象与值具有同样的表现, 但是会使复制的开销变得非常大, 在 C++ 文献中, 它被简称为 COW; 在 COW 的字符串中, 动态分配的内存可以在字符串间共享; 每个字符串都可以通过引用计数知道它们是否使用了共享内存; 当一个字符串被赋值给另一个字符串时, 所进行的处理只有复制指针以及增加引用计数。任何会改变字符串值的操作都会首先检查是否只有一个指针指向该字符串的内存; 如果多个字符串都指向该内存空间, 所有的变值操作(任何可能会改变字符串值的操作)都会在改变字符串值之前先分配新的内存空间并复制字符串. 写时复制这项技术太有名了, 以至于开发人员可能会想当然地以为 std::string 就是以这种方式实现的. 但是实际上, 写时复制甚至是不符合 C++11 标准的实现方式, 而且问题百出. 如果以写时复制方式实现字符串, 那么赋值和参数传递操作的开销很小, 但是一旦字符串被共享了, 非常量引用以及任何变值函数的调用都需要昂贵的分配和复制操作. 在并发代码中, 写时复制字符串的开销同样很大; 每次变值函数和非常量引用都要访问引用计数器; 当引用计数器被多个线程访问时, 每个线程都必须使用一个特殊的指令从主内存中得到引用计数的副本, 以确保没有其他线程改变这个值. C++11 及之后的版本中,随着"右值引用"和"移动语义"的出现, 使用它们可以在某种程度上减轻复制的负担; 如果一个函数使用"右值引用"作为参数, 那么当实参是一个右值表达式时, 字符串可以进行轻量级的指针复制, 从而节省一次复制操作.

### **5. 优化算法**
- 算法的时间开销是一个抽象的数学函数,它描述了随着输入数据规模的增加, 算法的时间开销会如何增长
- 时间开销通常使用大 O 标记表示, 例如 O(f(n))
- O(1), 即常量时间
- O(log2n), 时间开销比线性更小
- O(n), 即线性时间
- O(n log2n), 算法可能具有超线性时间开销
- O(n2)、O(n3), 对于有些问题，简单解决方案的时间开销是 O(n2) 或 O(n3), 而微妙一点的解决方案的速度会更快
- O(2n) 算法的时间开销增长得太快了, 它们应当只被应用于 n 很小的情况下
- 最优情况、平均情况和最差情况的时间开销; 通常的大 O 标记假设算法对任意输入数据集的运行时间是相同的;
- 当n很小时, 所有算法的时间开销都一样
- 经验丰富的开发人员不会只凭借自己独特的直觉去寻找改善性能的机会
- 用于改善性能的通用技巧, 它们非常实用: 数熟悉的数据结构、C++ 语言特性或是硬件创新的核心
    - 预计算: 在程序早期,例如设计时、编译时或是链接时, 通过在热点代码前执行计算来将计算从热点部分中移除
    - 延迟计算: 通过在真正需要执行计算时才执行计算, 可以将计算从某些代码路径上移除
    - 批量处理: 每次对多个元素一起进行计算, 而不是一次只对一个元素进行计算
    - 缓存: 通过保存和复用昂贵计算的结果来减少计算量, 而不是重复进行计算
    - 特化: 通过移除未使用的共性来减少计算量
    - 提高处理量: 通过一次处理一大组数据来减少循环处理的开销
    - 提示: 通过在代码中加入可能会改善性能的提示来减少计算量
    - 优化期待路径: 以期待频率从高到低的顺序对输入数据或是运行时发生的事件进行测试
    - 散列法: 计算可变长度字符串等大型数据结构的压缩数值映射(散列值), 在进行比较时用散列值代替数据结构可以提高性能
    - 双重检查: 通过先进行一项开销不大的检查, 然后只在必要时才进行另外一项开销昂贵的检查来减少计算量

> 如果没有必要在某个函数中的所有执行路径(if-then-else 逻辑的所有分支)上都进行计算,那就只在需要结果的路径上进行计算. 批量处理的目标是收集多份工作,然后一起处理它们; 批量处理可以用来移除重复的函数调用(缓存输出)或是每次只处理一个项目时会发生的其他计算. 提高处理量的目标是减少重复操作的迭代次数, 削减重复操作带来的开销; 向操作系统请求大量输入数据或是或发送大量输出数据, 来减少为少量内存块或是独立的数据项调用内核而产生的开销. 在有多个 else-if 分支的 if-then-else 代码块中, 如果条件语句的编写顺序是随机的, 那么每次执行经过 if-then-else 代码块时，都有大约一半的条件语句会被测试; 如果有一种情况的发生几率是 95%, 而且首先对它进行条件测试, 那么在 95% 的情况下都只会执行一次测试. 双重检查是指首先使用一种开销不大的检查来排除部分可能性, 然后在必要时再使用一个开销很大的检查来测试剩余的可能性; 双重检查常与缓存同时使用, 当处理器需要某个值时, 首先会去检查该值是否在缓存中; 如果不在, 则从内存中获取该值或是通过一项开销大的计算来得到该值.

### **6. 优化动态分配内存的变量**
- 除了使用非最优算法外, 乱用动态分配内存的变量就是 C++ 程序中最大的"性能杀手"
- 开发人员只要知道如何减少对内存管理器的调用就可以成为优秀的性能优化专家
- 从循环处理中或是会被频繁调用的函数中移除哪怕一次对内存管理器的调用,就能显著地改善性能,而且通常程序中有很多可被移除的调用
- 为变量分配内存的开销取决于存储期(生命周期): 静态存储期; 线程局部存储期; 自动存储期; 动态存储期; 
- 变量的所有权(ownership): 全局所有权; 词法作用域所有权; 成员所有权; 动态变量所有权;
- C++ 动态变量所有权对于性能优化非常重要, 具有强定义所有权的程序会比所有权分散的程序更高效
- 值对象与实体对象: 

> 有些变量通过它们的内容体现出它们在程序中的意义, 这些变量被称为值对象. 其他变量通过在程序中所扮演的角色体现出它们的意义, 这些变量被称为实体或实体对象. 一个变量是实体对象还是值对象决定了复制以及比较相等是否有意义; 实体不应当被复制和比较; 一个类的成员变量是实体还是值决定了应该如何编写该类的构造函数; 类实例可以共享实体的所有权, 但是无法有效地复制实体; 透彻地理解实体对象和值对象非常重要, 因为实体变量中往往包含许多动态分配内存的变量, 即使复制这些变量是合理的, 但其性能开销也是昂贵的.

- 指针抽象了**硬件的地址**来隐藏计算机架构的复杂性和变化性
- C++11中有一个称为 **nullptr** 的指针, 根据 C++ 标准, 它绝对不会指向有效的内存地址
- 未初始化的 C 风格的指针没有预定义值(出于性能考虑) NULL
- 当一个指向数组的指针被当作指向实例的指针删除时, 会导致未定义的行为, 反之亦然
- new 和 delete 表达式会调用 C++ 标准库的内存管理函数，在 C++ 标准中称为"自由存储区"的内存池中分配和归还内存
- 提供了经典的 C 函数库中的内存管理函数, 如用于分配和释放无类型的内存块的 malloc() 和 free()
- C++ 标准库提供了分配器模板, 它是 new 和 delete 表达式的泛化形式, 可以与标准容器一起使用

> 动态变量的所有权是由开发人员赋予, 并编码在程序逻辑中的. 当所有权很分散时, 每行代码都可能会创建出动态变量, 添加或是移除引用, 或是销毁变量; 开发人员必须追踪所有的执行路径, 确保动态变量总是正确地被返回给了内存管理器. 使用**智能指针**(类的 RAII 技术)实现动态变量**所有权**的自动化; 智能指针会通过耦合动态变量的生命周期与拥有该动态变量的智能指针的生命周期, 来实现动态变量所有权的自动化. 在通常情况下维护**一个所有者**,在特殊情况下使用 std::unique_ptr 维护所有权, 这样可以更加容易地判断一个动态变量是否指向一块有效的内存地址,以及当不再需要它时它是否会被正确地返回给内存管理器; 使用 unique_ptr 时会发生一些小的性能损失. 开发人员不能将 **C 风格**的指针(例如 new 表达式返回的指针)赋值给**多个**智能指针,而只能将其从一个智能指针赋值给另外一个智能指针; 如果将同一个 C 风格的指针赋值给多个智能指针, 那么该指针会被多次删除, 导致发生 C++ 标准中所谓的*"**未定义的行为**"; 这听起来很容易, 不过由于智能指针可以通过 C 风格的指针创建, 因此传递参数时所进行的**隐式的类型转换**会导致这种情况发生.

- 大多数 C++ 语句的开销都不过是几次内存访问而已; 不过, 为动态变量分配内存的开销则是数千次内存访问
- 哪怕只是移除一次对内存管理器的调用就可以带来**显著的性能提升**, 优化字符串例子中

> 从概念上说, 分配内存的函数会从**内存块集合**中寻找一块可以使用的内存来满足请求; 如果函数找到了一块正好符合大小的内存, 它会将这块内存从集合中移除并返回这块内存; 如果函数找到了一块比需求更大的内存,它可以选择拆分内存块然后只返回其中一部分; 显然这种描述为**许多实现**留下了可选择的空间. 如果没有可用的内存块来满足请求, 那么分配函数会**调用操作系统内核**, 从系统的可用内存池中请求额外的大块内存, 这次调用的开销非常大; 内核返回的内存可能会(也可能不会)被缓存在**物理 RAM** 中, 可能会导致初次访问时发生更大的延迟; 遍历可使用的内存块列表, 这一操作自身的开销也是昂贵的; 这些内存块分散在内存中, 而且与那些运行中的程序正在使用的内存块相比,它们也不太会被缓存起来. 未使用内存块的集合是由程序中的**所有线程所共享**的资源; 对未使用内存块的集合所进行的改变都必须是**线程安全**的; 如果若干个线程频繁地**调用内存管理器**分配内存或是释放内存, 那么它们会将内存管理器视为一种资源进行竞争, 导致除了一个线程外, 所有线程都必须等待. 当不再需要使用动态变量时, C++ 程序必须释放那些已经分配的内存; 从概念上说, 释放内存的函数会将内存块返回到**可用内存块集合**中; 在**实际的实现**中, 内存释放函数的行为会更复杂, 绝大多数实现方式都会尝试将**刚释放的内存块与临近的未使用的内存块合并**; 这样可以防止向未使用内存集合中放入过多太小的内存块(内存碎片化,也被认为是一种内存泄漏); 调用内存释放函数与调用内存分配函数有着相同的问题: 降低缓存效率和争夺对未使用的内存块的多线程访问.

- [环形缓冲区](https://www.boost.org/doc/libs/1_86_0/doc/html/circular_buffer.html)
- 减少动态变量的重新分配
    - 1. 预分配动态变量以防止重新分配 std::string std::vector 的成员函数reserve(size_t n)
    - std::string std::vector 的 shrink_to_fit() 成员函数将未使用的空间返回给内存管理器
    - 2. 在循环外创建动态变量 **04_loop_dynamic.cpp**
    - 3. 移除无谓的复制 开发人员在寻找一段热点代码中的优化机会时, 必须特别注意赋值和声明, 因为在这些地方可能会发生昂贵的复制(class and struct, NOT POD)
    - 4. 在类定义中禁止不希望发生的复制, delete and private
    - 5. 移除函数调用上的复制, 类对象-复制构造函数将会调用**内存管理器**(性能杀手)
    - 6. 移除函数返回上的复制, 类对象-复制构造通常都会发生实际的函数调用,类越大越复杂,时间开销也越大
    - 7. 调用方常常会像 auto res = scalar_product(arg_array, 10); 
    - 这样将函数返回值赋值给一个变量; 因此除了在函数内部调用复制构造外,
    - 在调用方还会调用复制构造函数或赋值运算符(有一次时间开销)
    - C++ 编译器(具体看实现)优化方法被称为复制省略(copy elision)或是返回值优化(return value optimization,RVO)
    - 这种方法就是不用 return 语句返回值,而是在函数内更新引用参数,然后通过输出参数返回该引用参数
    - 8. 免复制库: 当需要填充的缓冲区、结构体或其他数据结构是函数参数时, 传递引用穿越多层库调用的开销很小
    - 9. 实现写时复制惯用法: 当一个带有动态变量的对象被复制时, 也必须**复制该动态变量**, 这种复制被称为**深复制(deep copy)**; 通过复制指针, 而不是复制指针指向的变量得到包含无主指针的对象的副本, 这种复制被称为**浅复制(shallow copy)**
    - 10. 切割数据结构: 如 std::string_view
- 实现移动语义: 
    - 现代 C++ 提供标准方式来高效地将一个变量的内容移动到另一个变量中,避免那些不应当发生的复制开销    

page130