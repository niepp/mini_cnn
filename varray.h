#ifndef __VARRAY_H__
#define __VARRAY_H__

namespace mini_cnn
{

template<class T>
class _varray
{

public:
	_varray();
	_varray(int_t w, int_t h, int_t d, int_t n);
	_varray(int_t w, int_t h, int_t d);
	_varray(int_t w, int_t h);
	_varray(int_t w);
	~_varray();
	_varray(const _varray<T>&);
	_varray<T>& operator=(const _varray<T>&);

	void copy(const _varray<T>&);

	void reshape(int_t w, int_t h, int_t d, int_t n);
	void reshape(int_t w, int_t h, int_t d);
	void reshape(int_t w, int_t h);
	void reshape(int_t w);

	void resize(int_t w, int_t h, int_t d, int_t n);
	void resize(int_t w, int_t h, int_t d);
	void resize(int_t w, int_t h);
	void resize(int_t w);

	void make_zero();

	int_t dim() const;
	int_t size() const;
	int_t width() const;
	int_t height() const;
	int_t depth() const;
	int_t count() const;

	int_t arg_max() const;

	T& operator()(int_t w, int_t h, int_t d, int_t n);
	T& operator()(int_t w, int_t h, int_t d);
	T& operator()(int_t w, int_t h);
	T& operator()(int_t w);

	const T& operator()(int_t w, int_t h, int_t d, int_t n) const;
	const T& operator()(int_t w, int_t h, int_t d) const;
	const T& operator()(int_t w, int_t h) const;
	const T& operator()(int_t w) const;

	T& operator[](int_t idx);
	const T& operator[](int_t idx) const;

private:
	void _create(int_t w, int_t h, int_t d, int_t n);
	void _release();

private:
	int_t m_w;  // width
	int_t m_h;  // height
	int_t m_d;  // depth or channel
	int_t m_n;  // count
	T* m_data;
};

template <class T>
inline void _varray<T>::_create(int_t w, int_t h, int_t d, int_t n)
{
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = n;
	m_data = new T[w * h * d * n];
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
		delete[]m_data;
		m_data = nullptr;
	}
}

template <class T>
inline _varray<T>::_varray() : m_w(0), m_h(0), m_d(0), m_n(0)
{
	m_data = nullptr;
}

template <class T>
inline _varray<T>::_varray(int_t w, int_t h, int_t d, int_t n)
{
	_create(w, h, d, n);
}

template <class T>
inline _varray<T>::_varray(int_t w, int_t h, int_t d)
{
	_create(w, h, d, 1);
}

template <class T>
inline _varray<T>::_varray(int_t w, int_t h)
{
	_create(w, h, 1, 1);
}

template <class T>
inline _varray<T>::_varray(int_t w)
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
	int_t len = other.m_w * other.m_h * other.m_d * other.m_n;
	m_data = new T[len];
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
		delete[] m_data;
		m_data = nullptr;
	}

	m_w = other.m_w;
	m_h = other.m_h;
	m_d = other.m_d;
	m_n = other.m_n;
	int_t len = m_w * m_h * m_d * m_n;
	m_data = new T[len];
	::memcpy(m_data, other.m_data, len * sizeof(T));
	return *this;
}

template <class T>
inline void _varray<T>::copy(const _varray<T> &other)
{
	nn_assert(m_w == other.m_w);
	nn_assert(m_h == other.m_h);
	nn_assert(m_d == other.m_d);
	nn_assert(m_n == other.m_n);
	int_t len = m_w * m_h * m_d * m_n;
	::memcpy(m_data, other.m_data, len * sizeof(T));
}

template <class T>
inline void _varray<T>::reshape(int_t w, int_t h, int_t d, int_t n)
{
	nn_assert(m_w * m_h * m_d * m_n == w * h * d * n);
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = n;
}

template <class T>
inline void _varray<T>::reshape(int_t w, int_t h, int_t d)
{
	nn_assert(m_w * m_h * m_d * m_n == w * h * d);
	m_w = w;
	m_h = h;
	m_d = d;
	m_n = 1;
}

template <class T>
inline void _varray<T>::reshape(int_t w, int_t h)
{
	nn_assert(m_w * m_h * m_d * m_n == w * h);
	m_w = w;
	m_h = h;
	m_d = 1;
	m_n = 1;
}

template <class T>
inline void _varray<T>::reshape(int_t w)
{
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
inline void _varray<T>::resize(int_t w, int_t h, int_t d, int_t n)
{
	_release();
	_create(w, h, d, n);
}

template <class T>
inline void _varray<T>::resize(int_t w, int_t h, int_t d)
{
	_release();
	_create(w, h, d, 1);
}

template <class T>
inline void _varray<T>::resize(int_t w, int_t h)
{
	_release();
	_create(w, h, 1, 1);
}

template <class T>
inline void _varray<T>::resize(int_t w)
{
	_release();
	_create(w, 1, 1, 1);
}

template <class T>
inline int_t _varray<T>::dim() const
{
	return m_n > 1 ? 4 :
		(m_d > 1 ? 3 : (m_h > 1 ? 2 : (m_w > 0 ? 1 : 0)));
}

template <class T>
inline int_t _varray<T>::size() const
{
	return m_n * m_d * m_h * m_w;
}

template <class T>
inline int_t _varray<T>::width() const
{
	return m_w;
}

template <class T>
inline int_t _varray<T>::height() const
{
	return m_h;
}

template <class T>
inline int_t _varray<T>::depth() const
{
	return m_d;
}

template <class T>
inline int_t _varray<T>::count() const
{
	return m_n;
}

template <class T>
inline int_t _varray<T>::arg_max() const
{
	int_t sz = this->size();
	int_t max_idx = 0;
	T m = m_data[0];
	for (int_t i = 1; i < sz; ++i)
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
inline T& _varray<T>::operator()(int_t w, int_t h, int_t d, int_t n)
{
	nn_assert((n == 0 && dim() == 3) || dim() == 4);
	int_t maplen = m_w * m_h * m_d;
	return m_data[n * maplen + m_w * m_h * d + m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(int_t w, int_t h, int_t d)
{
	nn_assert((d == 0 && dim() == 2) || dim() == 3);
	return m_data[m_w * m_h * d + m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(int_t w, int_t h)
{
	nn_assert((h == 0 && dim() == 1) || dim() == 2);
	return m_data[m_w * h + w];
}

template <class T>
inline T& _varray<T>::operator()(int_t w)
{
	nn_assert(dim() == 1);
	return m_data[w];
}

template <class T>
inline const T& _varray<T>::operator()(int_t w, int_t h, int_t d, int_t n) const
{
	nn_assert((n == 0 && dim() == 3) || dim() == 4);
	int_t maplen = m_w * m_h * m_d;
	return m_data[n * maplen + m_w * m_h * d + m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(int_t w, int_t h, int_t d) const
{
	nn_assert((d == 0 && dim() == 2) || dim() == 3);
	return m_data[m_w * m_h * d + m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(int_t w, int_t h) const
{
	nn_assert((h == 0 && dim() == 1) || dim() == 2);
	return m_data[m_w * h + w];
}

template <class T>
inline const T& _varray<T>::operator()(int_t w) const
{
	nn_assert(dim() == 1);
	return m_data[w];
}

template <class T>
inline T& _varray<T>::operator[](int_t idx)
{
	nn_assert(dim() > 0);
	return m_data[idx];
}

template <class T>
inline const T& _varray<T>::operator[](int_t idx) const
{
	nn_assert(dim() > 0);
	return m_data[idx];
}

typedef _varray<float_t> varray;
typedef std::vector<varray*> varray_vec;

}

#endif // __VARRAY_H__

