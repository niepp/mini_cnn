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
void _MatrixMN<T>::SetColumn(unsigned long col, const _VectorN<T> &v)
{
	assert(col < _col && v.GetSize() == _row);
	for(unsigned long i = 0; i < _row; ++i)
	{
		operator()(i, col) = v[i];
	}
}

template <class T>
void _MatrixMN<T>::SetRow(unsigned long row, const _VectorN<T> &v)
{
	assert(row < _row && v.GetSize() == _col);
	for(unsigned long i = 0; i < _col; ++i)
	{
		operator()(row, i) = v[i];
	}
}

template <class T>
void _MatrixMN<T>::SetDiagonalEntries(const T &v)
{
	for(unsigned long i = 0; i < _row; i++)
	{
		operator()(i, i) = v;
	}
}

template <class T>
void _MatrixMN<T>::SetDiagonalEntries(const _VectorN<T> &v)
{
	assert(v.GetSize() >= rows);
	for(unsigned long i = 0; i < _row; i++)
	{
		operator()(i, i) = v[i];
	}
}

template <class T>
void _MatrixMN<T>::AddToDiagonal(const T &v)
{
	for(unsigned long i = 0; i < _row; ++i)
	{
		operator()(i, i) += v;
	}
}

template <class T>
void _MatrixMN<T>::MakeZero()
{
	memset(_buf, 0, _row * _col * sizeof(T));
}

template <class T>
void _MatrixMN<T>::MakeIdentity()
{
	SetDiagonalEntries((T)1);
}

template <class T>
_MatrixMN<T> _MatrixMN<T>::SubMatrix(unsigned long row1, unsigned long col1,
					   unsigned long row2, unsigned long col2) const
{
	assert(row1 <= row2 && col1 <= col2);
	unsigned long row = row2 - row1 + 1;
	unsigned long col = col2 - col1 + 1;
	_MatrixMN<T> m(row, col);
	for (unsigned long i = 0; i < row; ++i)
	{
		for (unsigned long j = 0; j < col; ++j)
		{
			m.operator()(i, j) = operator() (i + row1, j + col1);
		}
	}
	return m;
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

