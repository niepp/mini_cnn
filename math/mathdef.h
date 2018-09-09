#ifndef __MATH_DEF_H__
#define __MATH_DEF_H__

namespace mini_cnn
{
const Float cEPSILON = 1e-4;

#ifndef PI
const Float cPI = 3.141592653589793;
#endif

const Float cMIN_FLOAT = std::numeric_limits<Float>::min();
const Float cMAX_FLOAT = std::numeric_limits<Float>::max();

typedef _MatrixMN<Float> MatrixMN;
typedef _VectorN<Float> VectorN;
typedef _Matrix3D<Float> Matrix3D;
}
#endif // __MATH_DEF_H__
