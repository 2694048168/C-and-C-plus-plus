/**
 * @file ObserversWeather.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-26
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ ObserversWeather.cpp -std=c++20
 * clang++ ObserversWeather.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

// Demo2: 基于观察者模式实现的模拟天气预报
class Observer
{
public:
    virtual void update(float temperature, float humidity, float pressure) = 0;
};

class WeatherStation
{
private:
    float                   temperature;
    float                   humidity;
    float                   pressure;
    std::vector<Observer *> observers;

public:
    void registerObserver(Observer *observer)
    {
        observers.push_back(observer);
    }

    void removeObserver(Observer *observer) {}

    void notifyObservers()
    {
        for (Observer *observer : observers)
        {
            observer->update(temperature, humidity, pressure);
        }
    }

    void setMeasurements(float temp, float hum, float press)
    {
        temperature = temp;
        humidity    = hum;
        pressure    = press;
        notifyObservers();
    }
};

class Display : public Observer
{
public:
    void update(float temperature, float humidity, float pressure)
    {
        std::cout << " Display: Temperature = " << temperature << " °C, Humidity = " << humidity
                  << " %, Pressure = " << pressure << " hPa" << std::endl;
    }
};

// -----------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "------------------------------\n";

    WeatherStation weatherStation;
    Display        display1;
    Display        display2;
    weatherStation.registerObserver(&display1);
    weatherStation.registerObserver(&display2);
    weatherStation.setMeasurements(25.5, 60, 1013.2);
    weatherStation.setMeasurements(24.8, 58, 1014.5);

    std::cout << "------------------------------\n";

    return 0;
}
