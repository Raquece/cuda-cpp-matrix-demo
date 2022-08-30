#pragma once

#include <functional>

using namespace std;

template<typename T>
class matrix
{
public:
    T* DATA;
    int rows;
    int cols;
    int SIZE;
    int WIDTH;

    matrix(int ROWS, int COLS);
    void map(std::function<T(T, int, int)>);
    void init(T val);
    T getAt(int i);
    T getAt(int row, int col);
    void setAt(T val, int i);
    void setAt(T val, int row, int col);
    void free();
};

template<typename T>
matrix<T>::matrix(int ROWS, int COLS)
{
    rows = ROWS;
    cols = COLS;
    WIDTH = cols;
    SIZE = rows * cols;

    DATA = (T*)malloc(sizeof(T) * SIZE);
}

template<typename T>
void matrix<T>::init(T val)
{
    for (int i = 0; i < SIZE; i++)
    {
        DATA[i] = val;
    }
}

template<typename T>
void matrix<T>::map(std::function<T(T, int, int)> F)
{
    for (int i = 0; i < SIZE; i++)
    {
        int ROW = i / WIDTH;
        int COL = i % WIDTH;
        DATA[i] = F(DATA[i], ROW, COL);
    }
}

template<typename T>
T matrix<T>::getAt(int i)
{
    return DATA[i];
}

template<typename T>
T matrix<T>::getAt(int row, int col)
{
    return DATA[row * cols + col];
}

template<typename T>
void matrix<T>::setAt(T val, int i)
{
    DATA[i] = val;
}

template<typename T>
void matrix<T>::setAt(T val, int row, int col)
{
    DATA[row * cols + col] = val;
}

template<typename T>
void matrix<T>::free()
{
    std::free(DATA);
}
