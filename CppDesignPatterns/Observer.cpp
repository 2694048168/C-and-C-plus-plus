/**
 * @file Observer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <list>

struct WeatherData
{
    float temper = 0.f;
};

class InterfaceDisplay

{
public:
    virtual void Show(float temperature) = 0;

    virtual ~InterfaceDisplay() {}
};

class DisplayA : public InterfaceDisplay
{
    virtual void Show(float temperature)
    {
        std::cout << "DisplayA : " << temperature << '\n';
    }
};

class DisplayB : public InterfaceDisplay
{
    virtual void Show(float temperature)
    {
        std::cout << "DisplayB : " << temperature << '\n';
    }
};

class DisplayC : public InterfaceDisplay
{
    virtual void Show(float temperature)
    {
        std::cout << "DisplayC : " << temperature << '\n';
    }
};

// 对应稳定点, 抽象
// 对应变化点, 扩展(继承和组合)
class DataCenter
{
public:
    void Attach(InterfaceDisplay *ob)
    {
        obs.emplace_back(ob);
    }

    void Detach(InterfaceDisplay *ob)
    {
        obs.remove(ob);
    }

    void Notify()
    {
        float temper = CalcTemperature();
        for (const auto &ob : obs)
        {
            ob->Show(temper);
        }
    }

    // 接口隔离
private:
    WeatherData *GetWeatherData()
    {
        return new WeatherData;
    }

    float CalcTemperature()
    {
        WeatherData *data   = GetWeatherData();
        float        temper = data->temper;
        return temper;
    }

    // ? list, why not vector
    std::list<InterfaceDisplay *> obs;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    // 单例模式
    DataCenter *pCenter = new DataCenter;

    // 某个模块
    InterfaceDisplay *p_da = new DisplayA();
    pCenter->Attach(p_da);

    // ...
    InterfaceDisplay *p_db = new DisplayB();
    pCenter->Attach(p_db);
    // ...
    InterfaceDisplay *p_dc = new DisplayC();
    pCenter->Attach(p_dc);

    pCenter->Notify();
    // ------------------
    pCenter->Detach(p_db);
    pCenter->Notify();

    pCenter->Notify();

    return 0;
}
