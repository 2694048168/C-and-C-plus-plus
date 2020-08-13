#include "Matrix.hpp"
#include <fstream>

using namespace std;

template<typename elemType>
Matrix<elemType> operator+(const Matrix<elemType> &m1, const Matrix<elemType> &m2)
{
    Matrix<elemType> result(m1);
    result += m2;

    return result;
}

template<typename elemType>
Matrix<elemType> operator*(const Matrix<elemType> &m1, const Matrix<elemType> &m2)
{
    Matrix<elemType> result(m1.rows(), m2.cols());
    for (size_t ix = 0; ix < m1.rows(); ix++)
    {
        for (size_t jx = 0; jx < m1.cols(); jx++)
        {
            result(ix, jx) = 0;
            for (size_t kx = 0; kx < m1.cols(); kx++)
            {
                result(ix,jx) += m1(ix, kx) * m2(kx, jx);
            }
        }
    }
    return result;
}

template<typename elemType>
void Matrix<elemType>::operator+=(const Matrix &m)
{
    int matrix_size = cols() * rows();
    for (size_t ix = 0; ix < matrix_size; ++ix)
    {
        ( *(_matrix + ix )) += ( *(m._matrix + ix ));
    }
}

template<typename elemType>
ostream & Matrix<elemType>::print(ostream &os) const
{
    int col = cols();
    int matrix_size = col * rows();
    for (size_t ix = 0; ix < matrix_size; ++ix)
    {
        if (ix % col == 0)
        {
            os << endl;
        }
        os << ( *(_matrix + ix )) << ' ';
    }
    os << endl;

    return os;
}


int main(int argc, char*argv[])
{
    ofstream log("./log.txt");
    if (! log)
    {
        cerr << "can't open log file!" << endl;
        return -1;
    }

    Matrix<float> identity (4, 4);
    log << "identity: " << identity << endl;
    float ar[16] = {1., 0., 0., 0., 1., 0., 0.,
                    0., 0., 1., 0., 0., 0., 1. };

    for (size_t i = 0, k =  0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            identity(i, j) = ar[k++];
        }
    }
    log << "identity atfer set: " << identity << endl;

    Matrix<float> m(identity);
    log << "m: memberwise initialized: " << m << endl;

    Matrix<float> m2(8, 12);
    log << "m2: 8X12: " << m2 << endl;

    m2 = m;
    log << "m2 after memberwise assigned to m: " << m2 << endl;

    float ar2[16] = {1.3f, 0.4f, 2.6f, 8.2f, 6.2f, 1.7f, 1.3f, 8.3f,
                     4.2f, 7.4f, 2.7f, 1.9f, 6.3f, 8.1f, 5.6f, 6.6f	};
    
    Matrix<float> m3(4, 4);
    for (size_t ix = 0, kx = 0; ix < 4; ++ix)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            m3(ix, j) = ar2[kx++];
        }
    }
    log << "m3: assigned random value: " << m3 << endl;

    Matrix<float> m4 = m3 * identity;
    log << m4 << endl;

    Matrix<float> m5 = m3 + m4;
    log << m5 << endl;

    m3 += m4;
    log << m3 << endl;

    return 0;
}