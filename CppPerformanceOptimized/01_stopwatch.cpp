/**
 * @file 01_stopwatch.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

template<typename T>
class basic_stopwatch : T
{
    typedef typename T BaseTimer;

public:
    // 创建一个秒表, 开始计时一项程序活动（可选）
    explicit basic_stopwatch(bool start);
    explicit basic_stopwatch(const char *activity = "Stopwatch", bool start = true);
    basic_stopwatch(std::ostream &log, const char *activity = "Stopwatch", bool start = true);

    // 停止并销毁秒表
    ~basic_stopwatch();

    // 得到上一次计时时间（上一次停止时的时间）
    unsigned LapGet() const;

    // 判断：如果秒表正在运行，则返回true
    bool IsStarted() const;

    // 显示累计时间，一直运行，设置/返回上次计时时间
    unsigned Show(const char *event = "show");

    // 启动（重启）秒表，设置/返回上次计时时间
    unsigned Start(const char *event_name = "start");

    // 停止正在计时的秒表，设置/返回上次计时时间
    unsigned Stop(const char *event_name = "stop");

private:
    const char   *m_activity; // "activity"字符串
    unsigned      m_lap;      // 上次计时时间（上一次停止时的时间）
    std::ostream &m_log;      // 用于记录事件的流
};

/** stopwatch 的类型模板参数 T 的值的类是一个更加简单的计时器, 它提供了依赖于操作系统
 * 和 C++ 标准的函数去访问时标计数器. 多个版本的 TimerBase 类, 去测试各种不同的时标计数器的实现方式.
 * 在现代 C++ 处理器上, T 的值的类可以使用 C++ <chrono> 库, 或是可以直接从操作系统中得到时标.
 * 这种实现方式的优点是在不同操作系统之间具有可移植性, 但是它需要用到 C++11.
 */
class TimerBase
{
public:
    // 清除计时器
    TimerBase()
        : m_start(std::chrono::system_clock::time_point::min())
    {
    }

    // 清除计时器
    void Clear()
    {
        m_start = std::chrono::system_clock::time_point::min();
    }

    // 如果计时器正在计时，则返回true
    bool IsStarted() const
    {
        return (m_start.time_since_epoch() != std::chrono::system_clock::duration(0));
    }

    // 启动计时器
    void Start()
    {
        m_start = std::chrono::system_clock::now();
    }

    // 得到自计时开始后的毫秒值
    unsigned long GetMs()
    {
        if (IsStarted())
        {
            std::chrono::system_clock::duration diff;
            diff = std::chrono::system_clock::now() - m_start;
            return (unsigned)(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
        }
        return 0;
    }

private:
    std::chrono::system_clock::time_point m_start;
};

// 在 Windows 上和 Linux 上都可以使用的 clock() 函数
// 在不同 C++ 版本和不同操作系统之间具有可移植性,
// 缺点是在 Linux 上和 Windows 上, clock() 函数的测量结果略有不同.
class TimerBaseClock
{
public:
    // 清除计时器
    TimerBaseClock()
    {
        m_start = -1;
    }

    // 清除计时器
    void Clear()
    {
        m_start = -1;
    }

    // 如果计时器正在计时，则返回true
    bool IsStarted() const
    {
        return (m_start != -1);
    }

    // 启动计时器
    void Start()
    {
        m_start = clock();
    }

    // 得到自计时开始后的毫秒值
    unsigned long GetMs()
    {
        clock_t now;
        if (IsStarted())
        {
            now        = clock();
            clock_t dt = (now - m_start);
            return (unsigned long)(dt * 1000 / CLOCKS_PER_SEC);
        }
        return 0;
    }

private:
    clock_t m_start;
};

class Stopwatch
{
public:
    enum TimeFormat
    {
        NANOSECONDS,
        MICROSECONDS,
        MILLISECONDS,
        SECONDS
    };

    Stopwatch()
        : start_time()
        , laps({})
    {
        start();
    }

    ~Stopwatch() {}

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
        laps       = {start_time};
    }

    template<TimeFormat fmt = TimeFormat::MILLISECONDS>
    std::uint64_t lap()
    {
        const auto t      = std::chrono::high_resolution_clock::now();
        const auto last_r = laps.back();
        laps.push_back(t);
        return ticks<fmt>(last_r, t);
    }

    template<TimeFormat fmt = TimeFormat::MILLISECONDS>
    std::uint64_t elapsed()
    {
        const auto end_time = std::chrono::high_resolution_clock::now();
        return ticks<fmt>(start_time, end_time);
    }

    template<TimeFormat fmt_total = TimeFormat::MILLISECONDS, TimeFormat fmt_lap = fmt_total>
    std::pair<std::uint64_t, std::vector<std::uint64_t>> elapsed_laps()
    {
        std::vector<std::uint64_t> lap_times;
        lap_times.reserve(laps.size() - 1);

        for (std::size_t idx = 0; idx <= laps.size() - 2; idx++)
        {
            const auto lap_end   = laps[idx + 1];
            const auto lap_start = laps[idx];
            lap_times.push_back(ticks<fmt_lap>(lap_start, lap_end));
        }

        return {ticks<fmt_total>(start_time, laps.back()), lap_times};
    }

private:
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_pt;
    time_pt                                                             start_time;
    std::vector<time_pt>                                                laps;

    template<TimeFormat fmt = TimeFormat::MILLISECONDS>
    static std::uint64_t ticks(const time_pt &start_time, const time_pt &end_time)
    {
        const auto          duration = end_time - start_time;
        const std::uint64_t ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        switch (fmt)
        {
        case TimeFormat::NANOSECONDS:
        {
            return ns_count;
        }
        case TimeFormat::MICROSECONDS:
        {
            std::uint64_t up        = ((ns_count / 100) % 10 >= 5) ? 1 : 0;
            const auto    mus_count = (ns_count / 1000) + up;
            return mus_count;
        }
        case TimeFormat::MILLISECONDS:
        {
            std::uint64_t up       = ((ns_count / 100000) % 10 >= 5) ? 1 : 0;
            const auto    ms_count = (ns_count / 1000000) + up;
            return ms_count;
        }
        case TimeFormat::SECONDS:
        {
            std::uint64_t up      = ((ns_count / 100000000) % 10 >= 5) ? 1 : 0;
            const auto    s_count = (ns_count / 1000000000) + up;
            return s_count;
        }
        }
    }
};

constexpr Stopwatch::TimeFormat ns  = Stopwatch::TimeFormat::NANOSECONDS;
constexpr Stopwatch::TimeFormat mus = Stopwatch::TimeFormat::MICROSECONDS;
constexpr Stopwatch::TimeFormat ms  = Stopwatch::TimeFormat::MILLISECONDS;
constexpr Stopwatch::TimeFormat s   = Stopwatch::TimeFormat::SECONDS;

constexpr Stopwatch::TimeFormat nanoseconds  = Stopwatch::TimeFormat::NANOSECONDS;
constexpr Stopwatch::TimeFormat microseconds = Stopwatch::TimeFormat::MICROSECONDS;
constexpr Stopwatch::TimeFormat milliseconds = Stopwatch::TimeFormat::MILLISECONDS;
constexpr Stopwatch::TimeFormat seconds      = Stopwatch::TimeFormat::SECONDS;

std::string show_times(const std::vector<std::uint64_t> &times)
{
    std::string result("{");
    for (const auto &t : times)
    {
        result += std::to_string(t) + ",";
    }
    result.back() = static_cast<char>('}');
    return result;
}

// -------------------------------------
int main(int argc, const char *argv[])
{
    // stopwatch 类的最简单的用法用到了 RAII
    // Resource Acquisition Is Initialization, 资源获取就是初始化惯用法
    {
        TimerBase sw;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    {
        TimerBaseClock sw;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
