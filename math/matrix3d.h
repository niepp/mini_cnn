#ifndef __MATRIX_3D_H__
#define __MATRIX_3D_H__

#include <cassert>
namespace mini_cnn
{
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
	_Matrix3D(Int w, Int h, Int d);
	_Matrix3D(T* data, Int w, Int h, Int d);
	_Matrix3D(const _Matrix3D<T>&);
	~_Matrix3D();

	_Matrix3D<T>& operator=(const _Matrix3D<T>&);

	Int Width() const;
	Int Height() const;
	Int Depth() const;

	T& operator() (Int w, Int h, Int d);
	T  operator() (Int w, Int h, Int d) const;

	_Matrix3D<T>& operator+=(const _Matrix3D<T>&);
	_Matrix3D<T>& operator-=(const _Matrix3D<T>&);

	_Matrix3D<T>& operator*=(const T &scale);

	friend _Matrix3D<T> operator*(const _Matrix3D<T>&, const T &scale);

	_Matrix3D<T>& operator^=(const _Matrix3D<T>& other);

	_Matrix3D<T>& operator^(const _Matrix3D<T>& other);

	void MakeZero();

	_Matrix3D<T>& Copy(const _Matrix3D<T>&);

	void Conv(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
		Int stride_w, Int stride_h,
		Padding padding) const;

	void ConvDepthWise(std::vector<_Matrix3D<T>*> &retm, const _Matrix3D<T> &filter,
		Int stride_w, Int stride_h,
		Padding padding) const;

	void ConvDepthWise(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
		Int stride_w, Int stride_h,
		Padding padding) const;

	T SumByDepthWise(Int depth_idx) const;

	T Avg() const;

	void AddBias(const _VectorN<T>& bias);

	_VectorN<T>* Flatten() const;

	void DownSample(_Matrix3D<T> *retm, std::vector<Int> &idx_map,
		Int pool_w, Int pool_h,
		Int pool_stride_w, Int pool_stride_h) const;

	void UpSample(_Matrix3D<T> *retm, std::vector<Int> &idx_map,
		Int pool_w, Int pool_h,
		Int pool_stride_w, Int pool_stride_h) const;

private:
	T ConvByLocal(Int startw, Int starth, const _Matrix3D<T>& filter) const;

private:
	Int _w;
	Int _h;
	Int _d;
	T* _data;
};

template <class T>
inline _Matrix3D<T>::_Matrix3D(Int w, Int h, Int d) : _w(w), _h(h), _d(d)
{
	_data = new T[w * h * d];
	this->MakeZero();
}

template <class T>
inline _Matrix3D<T>::_Matrix3D(T* data, Int w, Int h, Int d) : _data(data), _w(w), _h(h), _d(d)
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
inline Int _Matrix3D<T>::Width() const
{
	return _w;
}

template <class T>
inline Int _Matrix3D<T>::Height() const
{
	return _h;
}

template <class T>
inline Int _Matrix3D<T>::Depth() const
{
	return _d;
}

template <class T>
inline T& _Matrix3D<T>::operator() (Int w, Int h, Int d)
{
	assert(w < _w && h < _h && d < _d);
	return _data[d * (_w * _h) + h * _w + w];
}

template <class T>
inline T _Matrix3D<T>::operator() (Int w, Int h, Int d) const
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
T _Matrix3D<T>::ConvByLocal(Int startw, Int starth, const _Matrix3D<T>& filter) const
{
	T s = 0;
	for (Int u = 0; u < filter._w; ++u)
	{
		Int x = startw + u;
		if (x < 0 || x >= _w)
		{
			continue;
		}
		for (Int v = 0; v < filter._h; ++v)
		{			
			Int y = starth + v;
			if (y < 0 || y >= _h)
			{
				continue;
			}
			for (Int c = 0; c < filter._d; ++c)
			{
				s += (*this)(x, y, c) * filter(u, v, c);
			}
		}
	}
	return s;
}

template <class T>
void _Matrix3D<T>::Conv(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters, 
	Int stride_w, Int stride_h, 
	Padding padding) const
{
	assert(filters.size() > 0 && filters[0] != nullptr);

	_Matrix3D<T> &filter = *filters[0];

	assert(_w >= filter._w && _h >= filter._h && _d == filter._d);

	Int nfilter = filters.size();

	// Padding::Valid
	Int nw = retm->_w;
	Int nh = retm->_h;
	for (Int i = 0; i < nw; ++i)
	{
		for (Int j = 0; j < nh; ++j)
		{
			Int startw = i * stride_w;
			Int starth = j * stride_h;
			for (Int k = 0; k < nfilter; ++k)
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
	Int stride_w, Int stride_h,
	Padding padding) const
{
	assert(retm.size() > 0 && retm[0] != nullptr);

	Int m = retm.size();

	assert(_w >= filter._w && _h >= filter._h && m == filter._d);

	// Padding::Valid
	Int nw = retm[0]->_w;
	Int nh = retm[0]->_h;
	for (Int k = 0; k < m; ++k)
	{
		_Matrix3D<T> &mat = *retm[k];
		for (Int i = 0; i < nw; ++i)
		{
			for (Int j = 0; j < nh; ++j)
			{
				Int startw = i * stride_w;
				Int starth = j * stride_h;

				for (Int c = 0; c < _d; ++c)
				{
					T s = 0;
					for (Int u = 0; u < filter._w; ++u)
					{
						Int x = startw + u;
						if (x < 0 || x >= _w)
						{
							continue;
						}
						for (Int v = 0; v < filter._h; ++v)
						{
							Int y = starth + v;
							if (y < 0 || y >= _h)
							{
								continue;
							}
							s += (*this)(x, y, c) * filter(u, v, k);
						}
					}
					mat(i, j, c) = s;
				}
			}
		}
	}
}

template <class T>
void _Matrix3D<T>::ConvDepthWise(_Matrix3D<T> *retm, const std::vector<_Matrix3D<T>*> &filters,
	Int stride_w, Int stride_h,
	Padding padding) const
{
	assert(filters.size() > 0 && filters[0] != nullptr);

	Int nfilter = filters.size();


	if (!(_w >= filters[0]->_w && _h >= filters[0]->_h)){
		int kkk = 0;
	}

	assert(_w >= filters[0]->_w && _h >= filters[0]->_h);

	assert(_d == nfilter && retm->_d == filters[0]->_d);

	// Padding::Valid
	Int nw = retm->_w;
	Int nh = retm->_h;
	Int nd = retm->_d;
	for (Int d = 0; d < nd; ++d)
	{
		for (Int i = 0; i < nw; ++i)
		{
			for (Int j = 0; j < nh; ++j)
			{
				Int startw = i * stride_w;
				Int starth = j * stride_h;
				T s = 0;
				for (Int k = 0; k < nfilter; ++k)
				{
					_Matrix3D<T> &filter = *filters[k];
					for (Int u = 0; u < filter._w; ++u)
					{
						Int x = startw - u;
						if (x < 0 || x >= _w)
						{
							continue;
						}
						for (Int v = 0; v < filter._h; ++v)
						{
							Int y = starth - v;
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
T _Matrix3D<T>::SumByDepthWise(Int depth_idx) const
{
	T s = 0;
	for (Int i = 0; i < _w; ++i)
	{
		for (Int j = 0; j < _h; ++j)
		{
			s += (*this)(i, j, depth_idx);
		}
	}
	return s;
}

template <class T>
T _Matrix3D<T>::Avg() const
{
	T avg = 0;
	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
			{
				avg += (*this)(i, j, k);
			}
		}
	}
	return avg / (_d * _w * _h);
}

template <class T>
void _Matrix3D<T>::AddBias(const _VectorN<T>& bias)
{
	assert(_d == bias.GetSize());
	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
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
void _Matrix3D<T>::DownSample(_Matrix3D<T> *retm, std::vector<Int> &idx_map,
	Int pool_w, Int pool_h,
	Int pool_stride_w, Int pool_stride_h) const
{

	assert(idx_map.size() == _w * _h);

	for (Int i = 0; i < idx_map.size(); ++i)
	{
		idx_map[i] = -1;
	}

	// Padding::Valid
	Int nw = retm->_w;
	Int nh = retm->_h;
	Int nd = retm->_d;
	for (Int d = 0; d < nd; ++d)
	{
		for (Int i = 0; i < nw; ++i)
		{
			for (Int j = 0; j < nh; ++j)
			{
				Int startw = i * pool_stride_w;
				Int starth = j * pool_stride_h;
				T maxv = cMIN_FLOAT;
				Int in_idx = -1;
				for (Int u = 0; u < pool_w; ++u)
				{
					Int x = startw + u;
					if (x < 0 || x >= _w)
					{
						continue;
					}
					for (Int v = 0; v < pool_h; ++v)
					{
						Int y = starth + v;
						if (y < 0 || y >= _h)
						{
							continue;
						}
						T s = (*this)(x, y, d);
						if (s > maxv)
						{
							maxv = s;
							in_idx = x + y * _w;
						}
					}
				}
				(*retm)(i, j, d) = maxv;
				if (in_idx >= 0)
				{
					Int out_idx = i + j * nw;
					idx_map[in_idx] = out_idx;
				}
			}
		}
	}

}


template <class T>
void _Matrix3D<T>::UpSample(_Matrix3D<T> *retm, std::vector<Int> &idx_map,
	Int pool_w, Int pool_h,
	Int pool_stride_w, Int pool_stride_h) const
{

	assert(idx_map.size() == retm->_w * retm->_h);

	// Padding::Valid
	Int nw = retm->_w;
	Int nh = retm->_h;
	Int nd = retm->_d;
	for (Int d = 0; d < nd; ++d)
	{
		for (Int i = 0; i < nw; ++i)
		{
			for (Int j = 0; j < nh; ++j)
			{
				Int in_idx = i + j * nw;
				Int out_idx = idx_map[in_idx];
				if (out_idx >= 0)
				{
					Int y = out_idx / _w;
					Int x = out_idx - y * _w;
					(*retm)(i, j, d) = (*this)(x, y, d);
				}
				else
				{
					(*retm)(i, j, d) = (T)(0);
				}
			}
		}
	}

}


template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator+=(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other._d);

	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
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

	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
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
	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
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
	_Matrix3D<T> *retMat = new _Matrix3D<T>(mat);
	for (Int k = 0; k < retMat._d; ++k)
	{
		for (Int i = 0; i < retMat._w; ++i)
		{
			for (Int j = 0; j < retMat._h; ++j)
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
	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
			{
				(*this)(i, j, k) *= other(i, j, k);
			}
		}
	}
	return *this;
}

// Hadamard product
template <class T>
_Matrix3D<T>& _Matrix3D<T>::operator^(const _Matrix3D<T>& other)
{
	assert(_w == other._w && _h == other._h && _d == other._d);
	_Matrix3D<T> *retMat = new _Matrix3D<T>(_w, _h, _d);
	for (Int k = 0; k < _d; ++k)
	{
		for (Int i = 0; i < _w; ++i)
		{
			for (Int j = 0; j < _h; ++j)
			{
				(*retMat)(i, j, k) = (*this)(i, j, k) * (*this)(i, j, k);
			}
		}
	}
	return *retMat;
}

}

#endif // __MATRIX_3D_H__

