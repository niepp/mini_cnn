#ifndef __MATRIX_3D_H__
#define __MATRIX_3D_H__

#include <cassert>

enum Padding
{
	Valid = 0,
	Same,
};

template<class T>
class _Matrix3D
{
public:
	_Matrix3D(size_t w, size_t h, size_t d);
	_Matrix3D(const _Matrix3D<T>&);
	~_Matrix3D();

	_Matrix3D<T>& operator=(const _Matrix3D<T>&);

	size_t Width() const;
	size_t Height() const;
	size_t Depth() const;

	T& operator() (size_t w, size_t h, size_t d);
	T  operator() (size_t w, size_t h, size_t d) const;

	void MakeZero();

	_Matrix3D<T>& Copy(const _Matrix3D<T>&);

	friend _Matrix3D<T> Conv(const _Matrix3D<T>& mat, const _Matrix3D<T>& filter, size_t stride_w, rsize_t stride_h, Padding padding);

private:
	size_t _w;
	size_t _h;
	size_t _d;
	T* _data;
};


template <class T>
inline _Matrix3D<T>::_Matrix3D(const _Matrix3D<T>& other) : _w(other._w), _h(other._h), _d(other._d)
{
	_data = new T[w * h * d];
	memcpy(_data, other._data, other._w * other._h * other._d * sizeof(T));
}

template <class T>
inline _Matrix3D<T>::~_Matrix3D()
{
	if (_data != nullptr)
	{
		delete[]_data;
		_data = nullptr;
	}
}

template <class T>
inline _Matrix3D<T>& _Matrix3D<T>::operator=(const _Matrix3D<T>& other)
{
	if (this == &other)
	{
		return *this;
	}

	if (_data != nullptr)
	{
		delete[] _data;
		_data = nullptr;
	}

	_w = other._w;
	_h = other._h;
	_d = other._d;
	_data = new T[_w * _h * _d];
	memcpy(_data, other._data, other._w * other._h * other._d * sizeof(T));
	return *this;
}

template <class T>
inline size_t _Matrix3D<T>::Width() const
{
	return _w;
}

template <class T>
inline size_t _Matrix3D<T>::Height() const
{
	return _h;
}

template <class T>
inline size_t _Matrix3D<T>::Depth() const
{
	return _d;
}

template <class T>
inline T& _Matrix3D<T>::operator() (size_t w, size_t h, size_t d)
{
	assert(w < _w && h < _h && d < _d);
	return _data[d * (_w * _h) + h * _w + w];
}

template <class T>
inline T _Matrix3D<T>::operator() (size_t w, size_t h, size_t d) const
{
	assert(w < _w && h < _h && d < _d);
	return _data[d * (_w * _h) + h * _w + w];
}

template <class T>
inline void _Matrix3D<T>::MakeZero()
{
	memset(_data, 0, _w * _h * _d * sizeof(T));
}

template <class T>
inline _Matrix3D<T>& _Matrix3D<T>::Copy(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other_d);
	memcpy(_data, other._data, other._w * other._h * other._d * sizeof(T));
	return *this;
}

template <class T>
_Matrix3D<T> Conv(const _Matrix3D<T>& mat, const _Matrix3D<T>& filter, size_t stride_w, rsize_t stride_h, Padding padding)
{

	assert(mat._w >= filter._w && mat._h >= filter._h && mat._d == filter._d);

	// Padding::Valid
	size_t nw = floorf(1.0f * (mat._w ¨C filter._w) / static_cast<float>(stride_w)) + 1;
	size_t nh = floorf(1.0f * (mat._h ¨C filter._h) / static_cast<float>(stride_h)) + 1;
	_Matrix3D<T> retm = new _Matrix3D<T>(nw, nh, 1);

	for (int i = 0; i < nw; ++i)
	{
		for (int j = 0; j < nh; ++j)
		{
			int startw = i * stride_w;
			int starth = j * stride_h;
			T s = 0;		
			for (int u = 0; u < filter._w; ++u)
			{
				for (int v = 0; v < filter._h; ++v)
				{	
					for (int c = 0; c < filter._d; ++c)
					{
						s += mat.operator(startw + u, starth + v, c) * filter.operator(u, v, c);
					}
				}
			}
			retm.operator(i, j, 0) = s;
		}
	}

	return retm;

}

#endif // __MATRIX_3D_H__

