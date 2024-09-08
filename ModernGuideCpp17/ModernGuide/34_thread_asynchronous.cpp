/**
 * @file 34_thread_asynchronous.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 多线程异步操作
 * @version 0.1
 * @date 2024-09-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 多线程异步操作
 * C++11中增加的线程类,使得能够非常方便的创建和使用线程,但有时会有些不方便,比如需要获取线程返回的结果,
 * 就不能通过join()得到结果,只能通过一些额外手段获得, 比如: 定义一个全局变量,在子线程中赋值,
 * 在主线程中读这个变量的值,整个过程比较繁琐. 
 * *C++提供的线程库中提供了一些类用于访问异步操作的结果.
 * ?主线程发起一个任务(子线程), 子线程执行任务过程中主线程去做别的事情了, 有两条时间线(异步);
 * ?主线程发起一个任务(子线程), 子线程执行任务过程中主线程没去做别的事情而是死等, 这时就只有一条时间线(同步),此时效率相对较低.
 * 因此多线程程序中的任务大都是异步的,主线程和子线程分别执行不同的任务,
 * 如果想要在主线中得到某个子线程任务函数返回的结果可以使用C++11提供的std:future类,这个类需要和其他类或函数搭配使用.
 * 
 * ===std:future 类的定义
 * 通过类的定义可以得知, future是一个模板类, 也就是这个类可以存储任意指定类型的数据;
// 定义于头文件 <future>
template< class T > class future;
template< class T > class future<T&>;
template<>          class future<void>;
// ======================================
 * ===std:future 构造函数
// ①
future() noexcept;
// ②
future( future&& other ) noexcept;
// ③
future( const future& other ) = delete;
// ========================================= 
 * 1. 构造函数①：默认无参构造函数;
 * 2. 构造函数②：移动构造函数，转移资源的所有权;
 * 3. 构造函数③：使用=delete显示删除拷贝构造函数, 不允许进行对象之间的拷贝;
 * ===std:future 常用成员函数（public)
 * *一般情况下使用=进行赋值操作就进行对象的拷贝，但是future对象不可用复制，因此会根据实际情况进行处理
future& operator=( future&& other ) noexcept;
future& operator=( const future& other ) = delete;
// =================================================
 * 1. 如果other是右值，那么转移资源的所有权;
 * 2. 如果other是非右值，不允许进行对象之间的拷贝（该函数被显示删除禁止使用）;
 * 
 * *取出future对象内部保存的数据, 其中void get()是为future<void>准备的,此时对象内部类型就是void,
 * 该函数是一个阻塞函数, 当子线程的数据就绪后解除阻塞就能得到传出的数值了.
T get();
T& get();
void get();
// =================================================
 * *因为future对象内部存储的是异步线程任务执行完毕后的结果, 是在调用之后的将来得到的,
 * 因此可以通过调用wait()方法, 阻塞当前线程, 等待这个子线程的任务执行完毕, 任务执行完毕当前线程的阻塞也就解除了.
void wait() const;

template< class Rep, class Period >
std::future_status wait_for( const std::chrono::duration<Rep,Period>& timeout_duration ) const;

template< class Clock, class Duration >
std::future_status wait_until( const std::chrono::time_point<Clock,Duration>& timeout_time ) const;
// =================================================
 * 如果当前线程wait()方法就会死等, 直到子线程任务执行完毕将返回值写入到future对象中,
 * 调用wait_for()只会让线程阻塞一定的时长, 但是这样并不能保证对应的那个子线程中的任务已经执行完毕了.
 * wait_until()和wait_for()函数功能是差不多, 前者是阻塞到某一指定的时间点, 后者是阻塞一定的时长;
 * ?当wait_until()和wait_for()函数返回之后,并不能确定子线程当前的状态,
 * 因此需要判断函数的返回值, 这样就能知道子线程当前的状态了:
 * 1. future_status::deferred 子线程中的任务函仍未启动;
 * 2. future_status::ready 子线程中的任务已经执行完毕, 结果已就绪;
 * 3. future_status::timeout 子线程中的任务正在执行中, 指定等待时长已用完;
 * 
 * *----------------------------------------
 * 2. std::promise
 * std::promise是一个协助线程赋值的类, 能够将数据和future对象绑定起来, 为获取线程函数中的某个值提供便利.
 * ===std:future 类的定义
 * 通过std::promise类的定义可以得知模板类,要在线程中传递什么类型的数据,模板参数就指定为什么类型.
// 定义于头文件 <future>
template< class R > class promise;
template< class R > class promise<R&>;
template<>          class promise<void>;
// ?==========================================
 * ===std:future 构造函数
// ①
promise();
// ②
promise( promise&& other ) noexcept;
// ③
promise( const promise& other ) = delete;
// ?=========================================
 * 1. 构造函数①：默认构造函数，得到一个空对象;
 * 2. 构造函数②：移动构造函数;
 * 3. 构造函数③：使用=delete显示删除拷贝构造函数, 不允许进行对象之间的拷贝;
 * ===std:future 公共成员函数
std::future<T> get_future();
// ?在std::promise类内部管理着一个future类对象,调用get_future()就可以得到这个future对象了.
// ?=========================================
void set_value( const R& value );
void set_value( R&& value );
void set_value( R& value );
void set_value();
// ?存储要传出的 value 值,并立即让状态就绪,这样数据被传出其它线程就可以得到这个数据了.
// ?重载的第四个函数是为promise<void>类型的对象准备的.
// ?=========================================
void set_value_at_thread_exit( const R& value );
void set_value_at_thread_exit( R&& value );
void set_value_at_thread_exit( R& value );
void set_value_at_thread_exit();
存储要传出的 value 值, 但是不立即令状态就绪. 在当前线程退出时,子线程资源被销毁,再令状态就绪.
// ?=========================================
 * ======std::promise的使用
 * 通过promise传递数据的过程一共分为5步:
 * 1. 在主线程中创建std::promise对象;
 * 2. 将这个std::promise对象通过引用的方式传递给子线程的任务函数;
 * 3. 在子线程任务函数中给std::promise对象赋值;
 * 4. 在主线程中通过std::promise对象取出绑定的future实例对象;
 * 5. 通过得到的future对象取出子线程任务函数中返回的值;
 * 
 * *----------------------------------------
 * 3. std::packaged_task
 * std::packaged_task类包装了一个可调用对象包装器类对象(可调用对象包装器包装的是可调用对象,可调用对象都可以作为函数来使用).
 * 这个类可以将内部包装的函数和future类绑定到一起,以便进行后续的异步调用,
 * 它和std::promise有点类似, std::promise内部保存一个共享状态的值,而std::packaged_task保存的是一个函数.
 * ======std::packaged_task 类的定义
// 定义于头文件 <future>
template< class > class packaged_task;
template< class R, class ...Args >
class packaged_task<R(Args...)>;
// ?=========================================
 * 通过类的定义可以看到这也是一个模板类,模板类型和要在线程中传出的数据类型是一致的
 * ======std::packaged_task 构造函数
// ①
packaged_task() noexcept;
// ②
template <class F>
explicit packaged_task( F&& f );
// ③
packaged_task( const packaged_task& ) = delete;
// ④
packaged_task( packaged_task&& rhs ) noexcept;
// ?=========================================
 * 1. 构造函数①：无参构造，构造一个无任务的空对象; 
 * 2. 构造函数②：通过一个可调用对象，构造一个任务对象; 
 * 3. 构造函数③：显示删除，不允许通过拷贝构造函数进行对象的拷贝;
 * 4. 构造函数④：移动构造函数;
 * ======std::packaged_task 常用公共成员函数
std::future<R> get_future();
// ?通过调用任务对象内部的get_future()方法就可以得到一个future对象,基于这个对象就可以得到传出的数据.
 * ======std::packaged_task 的使用
 *  packaged_task其实就是对子线程要执行的任务函数进行了包装,
 * 和可调用对象包装器的使用方法相同, 包装完毕之后直接将包装得到的任务对象传递给线程对象就可以.
 * 
 * *----------------------------------------
 * 4. std::async
 * std::async函数比前面提到的std::promise和packaged_task更高级一些,
 * ?因为通过这函数可以直接启动一个子线程并在这个子线程中执行对应的任务函数,
 * 异步任务执行完成返回的结果也是存储到一个future对象中,
 * 当需要获取异步任务的结果时, 只需要调用future 类的get()方法即可;
 * 如果不关注异步任务的结果, 只是简单地等待任务完成的话, 可以调用future 类的wait()或者wait_for()方法.
 * ======std::async 函数原型
// 定义于头文件 <future>
// ①
template< class Function, class... Args>
std::future<std::result_of_t<std::decay_t<Function>(std::decay_t<Args>...)>>
    async( Function&& f, Args&&... args );

// ②
template< class Function, class... Args >
std::future<std::result_of_t<std::decay_t<Function>(std::decay_t<Args>...)>>
    async( std::launch policy, Function&& f, Args&&... args );
// ?=========================================
 * 1. 函数①：直接调用传递到函数体内部的可调用对象，返回一个future对象;
 * 2. 函数②：通过指定的策略调用传递到函数内部的可调用对象，返回一个future对象;
 * *函数参数f：可调用对象，这个对象在子线程中被作为任务函数使用;
 * *函数参数Args：传递给 f 的参数（实参);
 * *函数参数policy：可调用对象·f的执行策略;
 * ======std::launch::async	调用async函数时创建新的线程执行任务函数;
 * ======std::launch::deferred 调用async函数时不执行任务函数,
 *    直到调用了future的get()或者wait()时才执行任务（这种方式不会创建新的线程）
 * 
 * 
 * 
 * 
 */

#include <future>
#include <iostream>
#include <thread>

// -----------------------------------
int main(int argc, const char **argv)
{
    // ?==========子线程任务函数执行期间，让状态就绪
    std::promise<int> pr;
    std::thread       t1(
        [](std::promise<int> &p)
        {
            p.set_value(100);
            std::this_thread::sleep_for(std::chrono::seconds(3));
            std::cout << "睡醒了....\n";
        },
        std::ref(pr));

    std::future<int> f     = pr.get_future();
    int              value = f.get();
    std::cout << "value: " << value << std::endl;

    t1.join();

    // *在外部主线程中创建的std::promise对象必须要通过引用的方式传递到子线程的任务函数中,
    // *在实例化子线程对象的时候,如果任务函数的参数是引用类型,
    // *那么实参一定要放到std::ref()函数中,表示要传递这个实参的引用到任务函数中.

    std::cout << "============================\n";
    // ?==========子线程任务函数执行结束，让状态就绪
    std::promise<int> pr2;
    std::thread       t2(
        [](std::promise<int> &p)
        {
            p.set_value_at_thread_exit(100);
            std::this_thread::sleep_for(std::chrono::seconds(3));
            std::cout << "睡醒了....\n";
        },
        std::ref(pr2));

    std::future<int> f2     = pr2.get_future();
    int              value2 = f2.get();
    std::cout << "value2: " << value2 << std::endl;

    t2.join();

    // ?==========std::packaged_task 使用方式
    std::cout << "============================\n";
    std::packaged_task<int(int)> task([](int x) { return x += 100; });

    std::thread t3(std::ref(task), 100);

    std::future<int> f3     = task.get_future();
    int              value3 = f3.get();
    std::cout << "value: " << value3 << std::endl;

    t3.join();

    // ?===========调用async()函数直接创建线程执行任务
    std::cout << "============================\n";
    std::cout << "主线程ID: " << std::this_thread::get_id() << std::endl;
    // 调用函数直接创建线程执行任务
    std::future<int> f4 = std::async(
        [](int x)
        {
            std::cout << "子线程ID: " << std::this_thread::get_id() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            return x += 100;
        },
        100);

    std::future_status status;
    do
    {
        status = f4.wait_for(std::chrono::seconds(1));
        if (status == std::future_status::deferred)
        {
            std::cout << "线程还没有执行...\n";
            f4.wait();
        }
        else if (status == std::future_status::ready)
        {
            std::cout << "子线程返回值: " << f4.get() << std::endl;
        }
        else if (status == std::future_status::timeout)
        {
            std::cout << "任务还未执行完毕, 继续等待...\n";
        }
    }
    while (status != std::future_status::ready);

    // ?===========调用async()函数不创建线程执行任务
    std::cout << "============================\n";
    // 调用函数直接创建线程执行任务
    std::future<int> f5 = async(
        std::launch::deferred,
        [](int x)
        {
            std::cout << "子线程ID: " << std::this_thread::get_id() << std::endl;
            return x += 100;
        },
        100);

    std::this_thread::sleep_for(std::chrono::seconds(5));
    // 由于指定了launch::deferred 策略,因此调用async()函数并不会创建新的线程执行任务,
    // 当使用future类对象调用了get()或者wait()方法后才开始执行任务.
    // !此处一定要注意调用wait_for()函数是不行的.
    std::cout << f5.get();

    //? 使用async()函数,是多线程操作中最简单的一种方式,不需要自己创建线程对象,并且可以得到子线程函数的返回值;
    //? 使用std::promise类, 在子线程中可以传出返回值也可以传出其他数据, 并且可选择在什么时机将数据从子线程中传递出来,使用起来更灵活;
    //? 使用std::packaged_task类, 可以将子线程的任务函数进行包装, 并且可以得到子线程的返回值;

    return 0;
}
