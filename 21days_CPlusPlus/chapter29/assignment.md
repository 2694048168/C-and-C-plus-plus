## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 29.8.1 测验
1. 我编写了一个图像处理应用程序，它在校正对比度时没有响应，我该如何办？
- 您的应用程序好像是在一个线程中执行所有的任务。因此，如果图像处理本身（调整对比度）是处理器密集型的， UI 将没有响应。应该将这两种操作放在两个线程中，这样操作系统将在这两个线程之间切换，向 UI 线程和执行对比度调整的工作线程提供处理器时间。

2. 我编写了一个多线程应用程序，能够以极快的速度访问数据库，但有时取回的数据不完整，请问是哪里出了问题？
- 可能没有妥善地同步线程。您的应用程序同时读写同一个对象，导致数据不一致。请使用一个二值信号量，在数据库表被访问时禁止对其进行修改。
