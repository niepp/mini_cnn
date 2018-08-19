#ifndef __MATRIX_MXN_H__
#define __MATRIX_MXN_H__

#include <cassert>

namespace mini_cnn
{
template <class T>
class _VectorN;

template <class T>
class _MatrixMN
{
public:
	_MatrixMN();
	_MatrixMN(unsigned long, unsigned long);
	_MatrixMN(unsigned long, unsigned long, T*);
	_MatrixMN(const _MatrixMN<T>&);
	~_MatrixMN();

	_MatrixMN<T>& operator=(const _MatrixMN<T>&);

	void SetSize(unsigned long, unsigned long);

	unsigned long GetRowCount() const;
	unsigned long GetColCount() const;

	T& operator() (unsigned long, unsigned  long);  
	T  operator() (unsigned long, unsigned  long) const;

	void MakeZero();

	_MatrixMN<T> Transpose() const;

	_MatrixMN<T>& Copy(const _MatrixMN<T>&);

	_VectorN<T> operator*(const _VectorN<T>&);
	friend _VectorN<T> operator*(const _VectorN<T>&, const _MatrixMN<T>&);

	_MatrixMN<T> operator*(const _MatrixMN<T>&);
	_MatrixMN<T> operator*(const T &scale);

	_MatrixMN<T> operator+(const _MatrixMN<T>&);
	_MatrixMN<T> operator-(const _MatrixMN<T>&);

	_MatrixMN<T> operator*=(const T &scale);

	_MatrixMN<T> operator+=(const _MatrixMN<T>&);
	_MatrixMN<T> operator-=(const _MatrixMN<T>&);

private:
	unsigned long _row, _col;
	T* _buf;
};

#include "matrixmxn.inl"
}
#endif // __MATRIX_MXN_H__