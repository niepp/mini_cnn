#ifndef __MATRIX_3D_H__
#define __MATRIX_3D_H__

#include <cassert>

enum Padding
{
	Valid = 0,
	Same,
};

template <class T>
class _VectorN;

template<class T>
class _Matrix3D
{
public:
	_Matrix3D(uint32_t w, uint32_t h, uint32_t d);
	_Matrix3D(const _Matrix3D<T>&);
	~_Matrix3D();

	_Matrix3D<T>& operator=(const _Matrix3D<T>&);

	uint32_t Width() const;
	uint32_t Height() const;
	uint32_t Depth() const;

	T& operator() (uint32_t w, uint32_t h, uint32_t d);
	T  operator() (uint32_t w, uint32_t h, uint32_t d) const;

	_Matrix3D<T>& operator+=(const _Matrix3D<T>&);
	_Matrix3D<T>& operator-=(const _Matrix3D<T>&);

	_Matrix3D<T>& operator*=(const T &scale);

	friend _Matrix3D<T> operator*(const _Matrix3D<T>&, const T &scale);

	_Matrix3D<T>& operator^(const _Matrix3D<T>& other);

	void MakeZero();

	_Matrix3D<T>& Copy(const _Matrix3D<T>&);

	void Conv(_Matrix3D<T>* retm, const std::vector<_Matrix3D<T>*> &filters, uint32_t stride_w, uint32_t stride_h, Padding padding) const;

	void AddBias(const _VectorN<T>& bias);

private:
	T ConvByLocal(uint32_t startw, uint32_t starth, const _Matrix3D<T>& filter) const;

private:
	uint32_t _w;
	uint32_t _h;
	uint32_t _d;
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
inline T& _Matrix3D<T>::operator() (uint32_t w, uint32_t h, uint32_t d)
{
	assert(w < _w && h < _h && d < _d);
	return _data[d * (_w * _h) + h * _w + w];
}

template <class T>
inline T _Matrix3D<T>::operator() (uint32_t w, uint32_t h, uint32_t d) const
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
	assert(_w == other._w && _h == other._h && _d == other._d);
	memcpy(_data, other._data, other._w * other._h * other._d * sizeof(T));
	return *this;
}

template <class T>
T _Matrix3D<T>::ConvByLocal(uint32_t startw, uint32_t starth, const _Matrix3D<T>& filter) const
{
	T s = 0;
	for (uint32_t u = 0; u < filter._w; ++u)
	{
		for (uint32_t v = 0; v < filter._h; ++v)
		{
			for (uint32_t c = 0; c < filter._d; ++c)
			{
				s += (*this)(startw + u, starth + v, c) * filter(u, v, c);
			}
		}
	}
	return s;
}

template <class T>
void _Matrix3D<T>::Conv(_Matrix3D<T>* retm, const std::vector<_Matrix3D<T>*> &filters, uint32_t stride_w, uint32_t stride_h, Padding padding) const
{
	assert(filters.size() > 0 && filters[0] != nullptr);

	_Matrix3D<T> &filter = *filters[0];

	assert(_w >= filter._w && _h >= filter._h && _d == filter._d);

	uint32_t nfilter = filters.size();

	// Padding::Valid
	uint32_t nw = retm->_w;
	uint32_t nh = retm->_h;
	for (uint32_t i = 0; i < nw; ++i)
	{
		for (uint32_t j = 0; j < nh; ++j)
		{
			uint32_t startw = i * stride_w;
			uint32_t starth = j * stride_h;
			for (uint32_t k = 0; k < nfilter; ++k)
			{
				const _Matrix3D<T>& filter = *filters[k];
				T s = ConvByLocal(startw, starth, filter);
				(*retm)(i, j, k) = s;
			}
		}
	}
}

template <class T>
void _Matrix3D<T>::AddBias(const _VectorN<T>& bias)
{
	assert(_d == bias.GetSize());
	for (uint32_t k = 0; k < _d; ++k)
	{
		for (uint32_t i = 0; i < _w; ++i)
		{
			for (uint32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) += bias[k];
			}
		}
	}
}

template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator+=(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other._d);

	for (uint32_t k = 0; k < _d; ++k)
	{
		for (uint32_t i = 0; i < _w; ++i)
		{
			for (uint32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) += other(i, j, k);
			}
		}
	}
	return *this;
}

template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator-=(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other._d);

	for (uint32_t k = 0; k < _d; ++k)
	{
		for (uint32_t i = 0; i < _w; ++i)
		{
			for (uint32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) -= other(i, j, k);
			}
		}
	}
	return *this;
}

template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator*=(const T &scale)
{
	for (uint32_t k = 0; k < _d; ++k)
	{
		for (uint32_t i = 0; i < _w; ++i)
		{
			for (uint32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) *= scale;
			}
		}
	}
	return *this;
}

template <class T>
inline _Matrix3D<T> operator*(const _Matrix3D<T>& mat, const T &scale)
{
	_Matrix3D<T> retMat = new _Matrix3D<T>(mat);
	for (uint32_t k = 0; k < retMat._d; ++k)
	{
		for (uint32_t i = 0; i < retMat._w; ++i)
		{
			for (uint32_t j = 0; j < retMat._h; ++j)
			{
				(*retMat)(i, j, k) *= scale;
			}
		}
	}
	return *retMat;
}

// Hadamard product
template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator^(const _Matrix3D<T>& other)
{	
	assert(_w == other._w && _h == other._h && _d == other._d);
	for (uint32_t k = 0; k < _d; ++k)
	{
		for (uint32_t i = 0; i < _w; ++i)
		{
			for (uint32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) *= other(i, j, k);
			}
		}
	}
	return *this;
}

#endif // __MATRIX_3D_H__

