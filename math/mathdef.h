#ifndef __MATH_DEF_H__
#define __MATH_DEF_H__

const float EPSILON = 1e-4f;

#ifndef PI
const float PI = 3.141592653589793f;
#endif

typedef _MatrixMN<float32_t> MatrixMN;
typedef _VectorN<float32_t> VectorN;
typedef _Matrix3D<float32_t> Matrix3D;

#endif // __MATH_DEF_H__
