#ifndef __VARRAY_H__
#define __VARRAY_H__

#pragma warning(disable:4316)

namespace mini_cnn
{

/*
	n(<=4) dimension array
	w * h * c * n

	************* --------
	*			*  |  	 |
	*			*  |  	 |
	*			* c=0    |
	*			*  |  	 |
	*			*  |  	 |
	************* -------|
	*			*  |  	 |
	*			*  |     |
	*			* c=1   n=0
	*			*  |   	 |
	*			*  |   	 |
	************* -------|
	*			*  |  	 |
	*			*  |  	 |
	*			* c=2    |
	*			*  |  	 |
	*			*  |  	 |
	************* --------
	*			*  |  	 |
	*			*  |  	 |
	*			* c=0    |
	*			*  |  	 |
	*			*  |  	 |
	************* -------|
	*			*  |  	 |
	*			*  |     |
	*			* c=1   n=1
	*			*  |   	 |
	*			*  |   	 |
	************* -------|
	*			*  |  	 |
	*			*  |  	 |
	*			* c=2    |
	*			*  |  	 |
	*			*  |  	 |
	************* --------
*/

template<class T>
class nn_align(nn_align_size) _varray
{
public:
	_varray();
	_varray(nn_int w, nn_int h, nn_int d, nn_int n);
	_varray(nn_int w, nn_int h, nn_int d);
	_varray(nn_int w, nn_int h);
	_varray(nn_int w);
	~_varray();
	_varray(const _varray<T>&);
	_varray<T>& operator=(const _varray<T>&);

	void copy(const _varray<T>&);

	void reshape(nn_int w, nn_int h, nn_int d, nn_int n);
	void reshape(nn_int w, nn_int h, nn_int d);
	void reshape(nn_int w, nn_int h);
	void reshape(nn_int w);

	void resize(nn_int w, nn_int h, nn_int d, nn_int n);
	void resize(nn_int w, nn_int h, nn_int d);
	void resize(nn_int w, nn_int h);
	void resize(nn_int w);

	void make_zero();

	nn_int dim() const;
	nn_int size() const;
	nn_int width() const;
	nn_int height() const;
	nn_int depth() const;
	nn_int count() const;

	nn_int arg_max() const;

	T& operator()(nn_int w, nn_int h, nn_int d, nn_int n);
	T& operator()(nn_int w, nn_int h, nn_int d);
	T& operator()(nn_int w, nn_int h);
	T& operator()(nn_int w);

	const T& operator()(nn_int w, nn_int h, nn_int d, nn_int n) const;
	const T& operator()(nn_int w, nn_int h, nn_int d) const;
	const T& operator()(nn_int w, nn_int h) const;
	const T& operator()(nn_int w) const;

	T& operator[](nn_int idx);
	const T& operator[](nn_int idx) const;

	bool check_dim(nn_int ndim) const;

private:
	void _create(nn_int w, nn_int h, nn_int d, nn_int n);
	void _release();

private:
	nn_int m_w;  // width
	nn_int m_h;  // height
	nn_int m_d;  // depth or channel
	nn_int m_n;  // count
	T* m_data;
};

template <class T>
inline void _varray<T>::_create(nn_int w, nn_int h, nn_int d, nn_int n)
{
	nn_assert(w >= 0 && h >= 0 && d >= 0 && n >= 0);
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = n;
	m_data = (T*)align_malloc(w * h * d * n * sizeof(T));
	this->make_zero();
}

template <class T>
inline void _varray<T>::_release()
{
	m_w = 0;
	m_h = 0;
	m_d = 0;
	m_n = 0;
	if (m_data != nullptr)
	{
		align_free(m_data);
		m_data = nullptr;
	}
}

template <class T>
inline _varray<T>::_varray() : m_w(0), m_h(0), m_d(0), m_n(0)
{
	m_data = nullptr;
}

template <class T>
inline _varray<T>::_varray(nn_int w, nn_int h, nn_int d, nn_int n)
{
	_create(w, h, d, n);
}

template <class T>
inline _varray<T>::_varray(nn_int w, nn_int h, nn_int d)
{
	_create(w, h, d, 1);
}

template <class T>
inline _varray<T>::_varray(nn_int w, nn_int h)
{
	_create(w, h, 1, 1);
}

template <class T>
inline _varray<T>::_varray(nn_int w)
{
	_create(w, 1, 1, 1);
}

template <class T>
inline _varray<T>::~_varray()
{
	_release();
}

template <class T>
inline _varray<T>::_varray(const _varray<T> &other) : m_w(other.m_w), m_h(other.m_h), m_d(other.m_d), m_n(other.m_n)
{
	nn_int len = other.m_w * other.m_h * other.m_d * other.m_n;
	m_data = (T*)align_malloc(len * sizeof(T));
	::memcpy(m_data, other.m_data, len * sizeof(T));
}

template <class T>
inline _varray<T>& _varray<T>::operator=(const _varray<T> &other)
{
	if (this == &other)
	{
		return *this;
	}

	if (m_data != nullptr)
	{
		align_free(m_data);
		m_data = nullptr;
	}

	m_w = other.m_w;
	m_h = other.m_h;
	m_d = other.m_d;
	m_n = other.m_n;
	nn_int len = m_w * m_h * m_d * m_n;
	m_data = (T*)align_malloc(len * sizeof(T));
	::memcpy(m_data, other.m_data, len * sizeof(T));
	return *this;
}

template <class T>
inline void _varray<T>::copy(const _varray<T> &other)
{
	nn_int len = m_w * m_h * m_d * m_n;
	nn_assert(len == other.m_w * other.m_h * other.m_d * other.m_n);
	::memcpy(m_data, other.m_data, len * sizeof(T));
}

template <class T>
inline void _varray<T>::reshape(nn_int w, nn_int h, nn_int d, nn_int n)
{
	nn_assert(w >= 0 && h >= 0 && d >= 0 && n >= 0);
	nn_assert(m_w * m_h * m_d * m_n == w * h * d * n);
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = n;
}

template <class T>
inline void _varray<T>::reshape(nn_int w, nn_int h, nn_int d)
{
	nn_assert(w >= 0 && h >= 0 && d >= 0);
	nn_assert(m_w * m_h * m_d * m_n == w * h * d);
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = 1;
}

template <class T>
inline void _varray<T>::reshape(nn_int w, nn_int h)
{
	nn_assert(w >= 0 && h >= 0);
	nn_assert(m_w * m_h * m_d * m_n == w * h);
	m_w = w;
	m_h = h;
	m_d = 1;
	m_n = 1;
}

template <class T>
inline void _varray<T>::reshape(nn_int w)
{
	nn_assert(w >= 0);
	nn_assert(m_w * m_h * m_d * m_n == w);
	m_w = w;
	m_h = 1;
	m_d = 1;
	m_n = 1;
}

template <class T>
inline void _varray<T>::make_zero()
{
	memset(m_data, 0, m_w * m_h * m_d * m_n * sizeof(T));
}

template <class T>
inline void _varray<T>::resize(nn_int w, nn_int h, nn_int d, nn_int n)
{
	_release();
	_create(w, h, d, n);
}

template <class T>
inline void _varray<T>::resize(nn_int w, nn_int h, nn_int d)
{
	_release();
	_create(w, h, d, 1);
}

template <class T>
inline void _varray<T>::resize(nn_int w, nn_int h)
{
	_release();
	_create(w, h, 1, 1);
}

template <class T>
inline void _varray<T>::resize(nn_int w)
{
	_release();
	_create(w, 1, 1, 1);
}

template <class T>
inline nn_int _varray<T>::dim() const
{
	return m_n > 1 ? 4 :
		(m_d > 1 ? 3 : (m_h > 1 ? 2 : (m_w > 0 ? 1 : 0)));
}

template <class T>
inline nn_int _varray<T>::size() const
{
	return m_n * m_d * m_h * m_w;
}

template <class T>
inline nn_int _varray<T>::width() const
{
	return m_w;
}

template <class T>
inline nn_int _varray<T>::height() const
{
	return m_h;
}

template <class T>
inline nn_int _varray<T>::depth() const
{
	return m_d;
}

template <class T>
inline nn_int _varray<T>::count() const
{
	return m_n;
}

template <class T>
inline nn_int _varray<T>::arg_max() const
{
	nn_int sz = this->size();
	nn_int max_idx = 0;
	T m = m_data[0];
	for (nn_int i = 1; i < sz; ++i)
	{
		if (m_data[i] > m)
		{
			m = m_data[i];
			max_idx = i;
		}
	}
	return max_idx;
}

template <class T>
inline T& _varray<T>::operator()(nn_int w, nn_int h, nn_int d, nn_int n)
{
	nn_assert(check_dim(4));
	nn_assert(w >= 0 && h >= 0 && d >= 0 && n >= 0);
	nn_assert(w < m_w && h < m_h && d < m_d && n < m_n);
	nn_int maplen = m_w * m_h * m_d;
	return m_data[n * maplen + m_w * m_h * d + m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(nn_int w, nn_int h, nn_int d)
{
	nn_assert(check_dim(3));
	nn_assert(w >= 0 && h >= 0 && d >= 0);
	nn_assert(w < m_w && h < m_h && d < m_d);
	return m_data[m_w * m_h * d + m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(nn_int w, nn_int h)
{
	nn_assert(check_dim(2));
	nn_assert(w >= 0 && h >= 0);
	nn_assert(w < m_w && h < m_h);
	return m_data[m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(nn_int w)
{
	nn_assert(check_dim(1));
	nn_assert(w >= 0);
	nn_assert(w < m_w);
	return m_data[w];
}

template <class T>
inline const T& _varray<T>::operator()(nn_int w, nn_int h, nn_int d, nn_int n) const
{
	nn_assert(check_dim(4));
	nn_assert(w >= 0 && h >= 0 && d >= 0 && n >= 0);
	nn_assert(w < m_w && h < m_h && d < m_d && n < m_n);
	nn_int maplen = m_w * m_h * m_d;
	return m_data[n * maplen + m_w * m_h * d + m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(nn_int w, nn_int h, nn_int d) const
{
	nn_assert(check_dim(3));
	nn_assert(w >= 0 && h >= 0 && d >= 0);
	nn_assert(w < m_w && h < m_h && d < m_d);
	return m_data[m_w * m_h * d + m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(nn_int w, nn_int h) const
{
	nn_assert(check_dim(2));
	nn_assert(w >= 0 && h >= 0);
	nn_assert(w < m_w && h < m_h);
	return m_data[m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(nn_int w) const
{
	nn_assert(check_dim(1));
	nn_assert(w < m_w);
	return m_data[w];
}

template <class T>
inline T& _varray<T>::operator[](nn_int idx)
{
	nn_assert(dim() > 0);
	nn_assert(idx >= 0 && idx < size());
	return m_data[idx];
}

template <class T>
inline const T& _varray<T>::operator[](nn_int idx) const
{
	nn_assert(dim() > 0);
	nn_assert(idx >= 0 && idx < size());
	return m_data[idx];
}

template <class T>
inline bool _varray<T>::check_dim(nn_int ndim) const
{
	nn_int _d = dim();
	if (ndim == 1){
		return _d == 1;
	}
	else if (ndim == 2){
		return _d == 1 || _d == 2;
	}
	else if (ndim == 3){
		return _d == 1 || _d == 2 || _d == 3;
	}
	else if (ndim == 4){
		return _d == 1 || _d == 2 || _d == 3 || _d == 4;
	}
	else{
		return false;
	}
}

typedef _varray<nn_float> varray;
typedef std::vector<varray*> varray_vec;

}

#endif // __VARRAY_H__

