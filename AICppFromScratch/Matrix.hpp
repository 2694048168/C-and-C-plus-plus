/**
 * @file Matrix.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Matrix implementation
 * @version 0.1
 * @date 2026-06-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace Ithaca {
class Matrix
{
private:
    std::vector<std::vector<double>> data;
    size_t                           rows;
    size_t                           cols;

public:
    Matrix(size_t r, size_t c)
        : rows(r)
        , cols(c)
    {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    size_t getRows() const
    {
        return rows;
    }

    size_t getCols() const
    {
        return cols;
    }

    double &operator()(size_t r, size_t c)
    {
        if (r >= getRows() || c >= getCols())
        {
            throw std::out_of_range("Index out of range");
        }

        return data[r][c];
    }

    const double &operator()(size_t r, size_t c) const
    {
        if (r >= getRows() || c >= getCols())
        {
            throw std::out_of_range("Index out of range");
        }

        return data[r][c];
    }

    Matrix &fillRandom(double min = -1.0, double max = 1.0)
    {
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(min, max);

        for (size_t i = 0; i < (*this).getRows(); ++i)
        {
            for (size_t j = 0; j < (*this).getCols(); ++j)
            {
                data[i][j] = dis(gen);
            }
        }

        return *this;
    }

    Matrix operator*(double scalar) const
    {
        Matrix result(getRows(), getCols());

        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = 0; j < getCols(); ++j)
            {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    // allowing in-place multiplication by scalar
    // allowing us to chain operators if needed
    Matrix operator*=(double scalar)
    {
        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = 0; j < getCols(); ++j)
            {
                data[i][j] *= scalar;
            }
        }
        return *this;
    }
};
} // namespace Ithaca
