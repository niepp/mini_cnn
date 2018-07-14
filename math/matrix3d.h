#ifndef __MATRIX_3D_H__
#define __MATRIX_3D_H__

#include <cassert>

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

#endif // __MATRIX_3D_H__
