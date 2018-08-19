#include "vectorn.h"

template <class T>
_MatrixMN<T>::_MatrixMN() : _row(0), _col(0), _buf(NULL)
{
}

template <class T>
_MatrixMN<T>::_MatrixMN(unsigned long row, unsigned long col) :_buf(NULL)
{
	SetSize(row, col);
	MakeZero();
}

template <class T>
_MatrixMN<T>::_MatrixMN(unsigned long row, unsigned long col, T* buf) :_buf(NULL)
{
	SetSize(row, col);
	memcpy(_buf, buf, row * col * sizeof(T));
}

template <class T>
_MatrixMN<T>::_MatrixMN(const _MatrixMN<T> &m) :_buf(NULL)
{
	SetSize(m._row, m._col);
	memcpy(_buf, m._buf, m._row * m._col * sizeof(T));
}

template <class T>
_MatrixMN<T>::~_MatrixMN()
{
	if(_buf != NULL)
	{
		delete []_buf;
		_buf = NULL;
	}
}

template <class T>
_MatrixMN<T>& _MatrixMN<T>::operator =(const _MatrixMN<T> &m)
{
	if (this == &m)
	{
		return *this;
	}

	if(_buf)
	{
		delete []_buf;
		_buf = NULL;
	}

	SetSize(m._row, m._col);
	memcpy(_buf, m._buf, m._row * m._col * sizeof(T));

	return *this;
}

template <class T>
T& _MatrixMN<T>::operator()(unsigned long row, unsigned long col)
{
	assert(row < _row && col < _col);
	return *(_buf + row * _col + col);
}

template <class T>
T _MatrixMN<T>::operator()(unsigned long row, unsigned long col) const
{
	assert(row < _row && col < _col);
	return *(_buf + row * _col + col);
}

template <class T>
unsigned long _MatrixMN<T>::GetRowCount() const
{
	return _row;
}

template <class T>
unsigned long _MatrixMN<T>::GetColCount() const
{
	return _col;
}

template <class T>
void _MatrixMN<T>::SetSize(unsigned long row, unsigned long col)
{
	_row = row;
	_col = col;

	if(_buf)
	{
		delete []_buf;
	}

	_buf = new T[_row * _col];
}

template <class T>
void _MatrixMN<T>::MakeZero()
{
	memset(_buf, 0, _row * _col * sizeof(T));
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::Transpose() const
{
	_MatrixMN<T> m(_col, _row);
	for(unsigned long i = 0; i < _row; ++i)
	{
		for(unsigned long j = 0; j < _col; ++j)
		{
			m(j, i) = operator()(i, j);
		}
	}

	return m;
}

template <class T>
_MatrixMN<T>& _MatrixMN<T>::Copy(const _MatrixMN<T>& other)
{
	assert(_row == other._row && _col == other._col);
	for (unsigned long i = 0; i < _row; ++i)
	{
		for (unsigned long j = 0; j < _col; ++j)
		{
			operator()(i, j) = other(i, j);
		}
	}
	return *this;
}

template <class T>
_VectorN<T> _MatrixMN<T>::operator*(const _VectorN<T> &v) 
{
	assert(v.GetSize() == _col);
	_VectorN<T> r(_row);
	for(unsigned long i = 0; i < _row; ++i)
	{
		for(unsigned long j = 0; j < _col; ++j)
		{
			r[i] += operator()(i,j) * v[j];
		}
	}

	return r;
}

template <class T>
_VectorN<T> operator*(const _VectorN<T> &v, const _MatrixMN<T> &m) 
{
	assert(v.GetSize() == m.row);
	_VectorN<T> r(m.col);
	for(unsigned long i = 0; i < m._col; i++ )
	{
		for(unsigned long j = 0; j < m._row; j++)
		{
			r[i] += m(j, i) * v[j];
		}
	}

	return r;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator*(const _MatrixMN<T> &r)
{
	assert(_col == r._row);
	_MatrixMN<T> result(_row, r._col);

	for(unsigned long i = 0; i < _row; i++)
	{
		for(unsigned long j = 0; j < r._col; j++)
		{
			for(unsigned long  k = 0; k < _col; k++)
			{
				result(i, j) += operator()(i, k) * r(k, j);
			}		
		}
	}

	return result;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator*(const T &scale)
{
	_MatrixMN<T> result(_row, _col);
	for(unsigned long i = 0; i < _row; i++)
	{
		for(unsigned long j = 0; j < _col; j++)
		{
			result(i, j) = operator()(i, j) * scale;
		}
	}
	return result;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator+(const _MatrixMN<T> &other)
{
	assert(_row == other._row && _col == other._col);
	_MatrixMN<T> result(_row, _col);

	for(unsigned long i = 0; i < _row; i++)
	{
		for(unsigned long j = 0; j < _col; j++)
		{
			result(i, j) = operator()(i, j) + other(i, j);
		}
	}

	return result;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator-(const _MatrixMN<T> &other)
{
	assert(_row == other._row && _col == other._col);
	_MatrixMN<T> result(_row, _col);

	for(unsigned long i = 0; i < _row; i++)
	{
		for(unsigned long j = 0; j < _col; j++)
		{
			result(i, j) = operator()(i, j) - other(i, j);
		}
	}

	return result;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator*=(const T &scale)
{
	assert(this->_row == other._row && this->_col == other._col);
	for (unsigned long i = 0; i < _row; i++)
	{
		for (unsigned long j = 0; j < _col; j++)
		{
			operator()(i, j) *= scale;
		}
	}
	return *this;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator+=(const _MatrixMN<T> &other)
{
	assert(this->_row == other._row && this->_col == other._col);
	for (unsigned long i = 0; i < _row; ++i)
	{
		for (unsigned long j = 0; j < _col; ++j)
		{
			operator()(i, j) += other(i, j);
		}
	}
	return *this;
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::operator-=(const _MatrixMN<T> &other)
{
	assert(this->_row == other._row && this->_col == other._col);
	for (unsigned long i = 0; i < _row; ++i)
	{
		for (unsigned long j = 0; j < _col; ++j)
		{
			operator()(i, j) -= other(i, j);
		}
	}
	return *this;
}
