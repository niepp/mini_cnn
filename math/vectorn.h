#ifndef __VECTOR_N_H__
#define __VECTOR_N_H__

#include <cassert>

namespace mini_cnn
{

template<class T>
class _Matrix3D;

template <class T>
class _MatrixMN;

template <class T>
class _VectorN
{
public:
	_VectorN() : _size(0), _buf(NULL)
	{
	}

	_VectorN(unsigned long size) : _size(size)
	{
		_buf = new T[size];
		::memset(_buf, 0, size * sizeof(T));
	}

	_VectorN(unsigned long size, T v) : _size(size)
	{
		_buf = new T[size];
		for(unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] = v;
		}
	}

	_VectorN(T* buf, Int size) : _size(size), _buf(buf)
	{
	}

	_VectorN(const _VectorN<T> &src)
	{
		_size = src._size;
		_buf = new T[src._size];
		for(unsigned long i = 0; i < src._size; ++i)
		{
			_buf[i] = src._buf[i];
		}
	}

	~_VectorN()
	{
		if(_buf != NULL)
		{
			delete[] _buf;
			_buf = NULL;
		}
	}

	_VectorN<T>& operator=(const _VectorN<T>& src)
	{
		if(this == &src)
		{	
			return *this;
		}

		if(_buf != NULL)
		{
			delete[] _buf;
		}

		_size = src._size;
		_buf = new T[src._size];
		for(unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] = src._buf[i];
		}

		return *this;
	}

	void SetSize(unsigned long size)
	{
		_size = size;
		_buf = new T[size];
		MakeZero();
	}

	void MakeZero()
	{
		::memset(_buf, 0, _size * sizeof(T));
	}

	T SquareMagnitude() const
	{
		T len2 = 0;
		for(unsigned long i = 0; i < _size; ++i)
		{
			len2 += _buf[i] * _buf[i];
		}
		return len2;
	}

	T Avg() const
	{
		T avg = 0;
		for (unsigned long i = 0; i < _size; ++i)
		{
			avg += _buf[i];
		}
		return avg / _size;
	}

	void Sqrt()
	{
		for (unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] = sqrt(_buf[i]);
		}
	}

	T Dot(const _VectorN<T>& src) const
	{
		T dot = 0;
		assert(_size == src._size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			dot += _buf[i] * src._buf[i];
		}
		return dot;
	}

	_VectorN<T>& Copy(const _VectorN<T>& src)
	{
		assert(_size == src._size);
		for (unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] = src._buf[i];
		}
		return *this;
	}

	_Matrix3D<T>* Unflatten(Int w, Int h, Int d) const
	{
		assert(_size == w * h * d);
		return new _Matrix3D<T>(_buf, w, h, d);
	}

	T& operator[](unsigned long index)
	{
		return _buf[index];
	}

	T operator[](unsigned long index) const
	{
		return _buf[index];
	}

	unsigned long GetSize() const
	{
		return _size;
	}

	_VectorN<T>& operator*=(const T &scale)
	{
		for(unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] *= scale;
		}
		return *this;
	}

	_VectorN<T>& operator+=(const _VectorN<T>&other)
	{
		assert(_size == other._size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] += other._buf[i];
		}
		return *this;
	}

	_VectorN<T>& operator-=(const _VectorN<T>&other)
	{
		assert(_size == other._size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			_buf[i] -= other._buf[i];
		}
		return *this;
	}

	_VectorN<T> operator*(const T &scale) const
	{
		_VectorN<T> result(_size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			result._buf[i] = _buf[i] * scale;
		}
		return result;
	}

	_VectorN<T> operator+(const _VectorN<T>&other) const
	{
		assert(_size == other._size);
		_VectorN<T> result(_size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			result._buf[i] = _buf[i] + other._buf[i];
		}
		return result;
	}

	_VectorN<T> operator-(const _VectorN<T>&other) const
	{
		assert(_size == other._size);
		_VectorN<T> result(_size);
		for(unsigned long i = 0; i < _size; ++i)
		{
			result._buf[i] = _buf[i] - other._buf[i];
		}
		return result;
	}

	// Hadamard product
	_VectorN<T> operator^(const _VectorN<T>&other) const
	{
		assert(_size == other._size);
		_VectorN<T> result(_size);
		for (unsigned long i = 0; i < _size; ++i)
		{
			result._buf[i] = _buf[i] * other._buf[i];
		}
		return result;
	}

	_MatrixMN<T> operator*(const _VectorN<T>&other) const
	{
		_MatrixMN<T> result(_size, other._size);
		for (unsigned long i = 0; i < _size; ++i)
		{
			for (unsigned long j = 0; j < other._size; ++j)
			{
				result(i, j) = _buf[i] * other._buf[j];
			}
		}
		return result;
	}

	uInt ArgMax() const
	{
		assert(_size > 0);
		uInt idx = 0;
		T maxv = _buf[0];
		for (unsigned long i = 1; i < _size; ++i)
		{
			if (_buf[i] > maxv)
			{
				maxv = _buf[i];
				idx = i;
			}
		}
		return idx;
	}

	uInt ArgMin() const
	{
		assert(_size > 0);
		uInt idx = 0;
		T minv = _buf[0];
		for (unsigned long i = 1; i < _size; ++i)
		{
			if (_buf[i] < minv)
			{
				minv = _buf[i];
				idx = i;
			}
		}
		return idx;
	}

private:
	unsigned long _size;
	T* _buf;
};
}

#endif // __VECTOR_N_H__