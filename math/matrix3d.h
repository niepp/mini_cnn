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
	_Matrix3D(int32_t w, int32_t h, int32_t d);
	_Matrix3D(T* data, int32_t w, int32_t h, int32_t d);
	_Matrix3D(const _Matrix3D<T>&);
	~_Matrix3D();

	_Matrix3D<T>& operator=(const _Matrix3D<T>&);

	int32_t Width() const;
	int32_t Height() const;
	int32_t Depth() const;

	T& operator() (int32_t w, int32_t h, int32_t d);
	T  operator() (int32_t w, int32_t h, int32_t d) const;

	_Matrix3D<T>& operator+=(const _Matrix3D<T>&);
	_Matrix3D<T>& operator-=(const _Matrix3D<T>&);

	_Matrix3D<T>& operator*=(const T &scale);

	friend _Matrix3D<T> operator*(const _Matrix3D<T>&, const T &scale);

	_Matrix3D<T>& operator^=(const _Matrix3D<T>& other);

	void MakeZero();

	_Matrix3D<T>& Copy(const _Matrix3D<T>&);

	void Conv(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
		int32_t stride_w, int32_t stride_h,
		Padding padding) const;

	void AddBias(const _VectorN<T>& bias);

	void ConvDepthWise(std::vector<_Matrix3D<T>*> &retm, const _Matrix3D<T> &filter,
		int32_t stride_w, int32_t stride_h,
		Padding padding) const;

	void ConvDepthWise(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
		int32_t stride_w, int32_t stride_h,
		Padding padding) const;

	T SumByDepthWise(int32_t depth_idx) const;

	_VectorN<T>* Flatten() const;

private:
	T ConvByLocal(int32_t startw, int32_t starth, const _Matrix3D<T>& filter) const;

private:
	int32_t _w;
	int32_t _h;
	int32_t _d;
	T* _data;
};

template <class T>
inline _Matrix3D<T>::_Matrix3D(int32_t w, int32_t h, int32_t d) : _w(w), _h(h), _d(d)
{
	_data = new T[w * h * d];
	this->MakeZero();
}

template <class T>
inline _Matrix3D<T>::_Matrix3D(T* data, int32_t w, int32_t h, int32_t d) : _data(data), _w(w), _h(h), _d(d)
{
}

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
inline int32_t _Matrix3D<T>::Width() const
{
	return _w;
}

template <class T>
inline int32_t _Matrix3D<T>::Height() const
{
	return _h;
}

template <class T>
inline int32_t _Matrix3D<T>::Depth() const
{
	return _d;
}

template <class T>
inline T& _Matrix3D<T>::operator() (int32_t w, int32_t h, int32_t d)
{
	assert(w < _w && h < _h && d < _d);
	return _data[d * (_w * _h) + h * _w + w];
}

template <class T>
inline T _Matrix3D<T>::operator() (int32_t w, int32_t h, int32_t d) const
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
T _Matrix3D<T>::ConvByLocal(int32_t startw, int32_t starth, const _Matrix3D<T>& filter) const
{
	T s = 0;
	for (int32_t u = 0; u < filter._w; ++u)
	{
		int32_t x = startw + u;
		if (x < 0 || x >= _w)
		{
			continue;
		}
		for (int32_t v = 0; v < filter._h; ++v)
		{			
			int32_t y = starth + v;
			if (y < 0 || y >= _h)
			{
				continue;
			}
			for (int32_t c = 0; c < filter._d; ++c)
			{
				s += (*this)(x, y, c) * filter(u, v, c);
			}
		}
	}
	return s;
}

template <class T>
void _Matrix3D<T>::Conv(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters, 
	int32_t stride_w, int32_t stride_h, 
	Padding padding) const
{
	assert(filters.size() > 0 && filters[0] != nullptr);

	_Matrix3D<T> &filter = *filters[0];

	assert(_w >= filter._w && _h >= filter._h && _d == filter._d);

	int32_t nfilter = filters.size();

	// Padding::Valid
	int32_t nw = retm->_w;
	int32_t nh = retm->_h;
	for (int32_t i = 0; i < nw; ++i)
	{
		for (int32_t j = 0; j < nh; ++j)
		{
			int32_t startw = i * stride_w;
			int32_t starth = j * stride_h;
			for (int32_t k = 0; k < nfilter; ++k)
			{
				const _Matrix3D<T>& filter = *filters[k];
				T s = ConvByLocal(startw, starth, filter);
				(*retm)(i, j, k) = s;
			}
		}
	}
}

template <class T>
void _Matrix3D<T>::ConvDepthWise(std::vector<_Matrix3D<T>*> &retm, const _Matrix3D<T> &filter,
	int32_t stride_w, int32_t stride_h,
	Padding padding) const
{
	assert(retm.size() > 0 && retm[0] != nullptr);

	int32_t m = retm.size();

	assert(_w >= filter._w && _h >= filter._h && m == filter._d);

	// Padding::Valid
	int32_t nw = retm[0]->_w;
	int32_t nh = retm[0]->_h;
	for (int32_t k = 0; k < m; ++k)
	{
		_Matrix3D<T> &mat = *retm[k];
		for (int32_t i = 0; i < nw; ++i)
		{
			for (int32_t j = 0; j < nh; ++j)
			{
				int32_t startw = i * stride_w;
				int32_t starth = j * stride_h;

				for (int32_t c = 0; c < _d; ++c)
				{
					T s = 0;
					for (int32_t u = 0; u < filter._w; ++u)
					{
						int32_t x = startw + u;
						if (x < 0 || x >= _w)
						{
							continue;
						}
						for (int32_t v = 0; v < filter._h; ++v)
						{
							int32_t y = starth + v;
							if (y < 0 || y >= _h)
							{
								continue;
							}
							s += (*this)(x, y, c) * filter(u, v, k);
							mat(i, j, c) = s;
						}
					}
				}
			}
		}
	}
}

template <class T>
void _Matrix3D<T>::ConvDepthWise(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
	int32_t stride_w, int32_t stride_h,
	Padding padding) const
{
	assert(filters.size() > 0 && filters[0] != nullptr);

	int32_t nfilter = filters.size();

	assert(_w >= filters[0]->_w && _h >= filters[0]->_h);

	assert(_d == nfilter && retm->_d == filters[0]->_d);

	// Padding::Valid
	int32_t nw = retm->_w;
	int32_t nh = retm->_h;
	int32_t nd = retm->_d;
	for (int32_t d = 0; d < nd; ++d)
	{
		for (int32_t i = 0; i < nw; ++i)
		{
			for (int32_t j = 0; j < nh; ++j)
			{
				int32_t startw = i * stride_w;
				int32_t starth = j * stride_h;
				T s = 0;
				for (int32_t k = 0; k < nfilter; ++k)
				{
					_Matrix3D<T> &filter = *filters[k];					
					for (int32_t u = 0; u < filter._w; ++u)
					{
						int32_t x = startw - u;
						if (x < 0 || x >= _w)
						{
							continue;
						}
						for (int32_t v = 0; v < filter._h; ++v)
						{
							int32_t y = starth - v;
							if (y < 0 || y >= _h)
							{
								continue;
							}
							s += (*this)(x, y, k) * filter(u, v, d);
						}
					}
				}
				(*retm)(i, j, d) = s;
			}
		}
	}
}

template <class T>
T _Matrix3D<T>::SumByDepthWise(int32_t depth_idx) const
{
	T s = 0;
	for (int32_t i = 0; i < _w; ++i)
	{
		for (int32_t j = 0; j < _h; ++j)
		{
			s += (*this)(i, j, depth_idx);
		}
	}
	return s;
}

template <class T>
void _Matrix3D<T>::AddBias(const _VectorN<T>& bias)
{
	assert(_d == bias.GetSize());
	for (int32_t k = 0; k < _d; ++k)
	{
		for (int32_t i = 0; i < _w; ++i)
		{
			for (int32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) += bias[k];
			}
		}
	}
}

template <class T>
_VectorN<T>* _Matrix3D<T>::Flatten() const
{
	return new _VectorN<T>(_data, _w * _h * _d);
}

template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator+=(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other._d);

	for (int32_t k = 0; k < _d; ++k)
	{
		for (int32_t i = 0; i < _w; ++i)
		{
			for (int32_t j = 0; j < _h; ++j)
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

	for (int32_t k = 0; k < _d; ++k)
	{
		for (int32_t i = 0; i < _w; ++i)
		{
			for (int32_t j = 0; j < _h; ++j)
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
	for (int32_t k = 0; k < _d; ++k)
	{
		for (int32_t i = 0; i < _w; ++i)
		{
			for (int32_t j = 0; j < _h; ++j)
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
	for (int32_t k = 0; k < retMat._d; ++k)
	{
		for (int32_t i = 0; i < retMat._w; ++i)
		{
			for (int32_t j = 0; j < retMat._h; ++j)
			{
				(*retMat)(i, j, k) *= scale;
			}
		}
	}
	return *retMat;
}

// Hadamard product
template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator^=(const _Matrix3D<T>& other)
{	
	assert(_w == other._w && _h == other._h && _d == other._d);
	for (int32_t k = 0; k < _d; ++k)
	{
		for (int32_t i = 0; i < _w; ++i)
		{
			for (int32_t j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) *= other(i, j, k);
			}
		}
	}
	return *this;
}

#endif // __MATRIX_3D_H__

