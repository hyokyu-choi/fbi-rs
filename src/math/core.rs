use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

pub trait LinearSpace:
    Sized
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    // + Mul<Self, Output = Self>
    // + Div<Self, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + fmt::Display
    + fmt::Debug
{
    type Data;

    fn new(data: Self::Data) -> Self;
    fn zero() -> Self;
    fn size(&self) -> usize;
    fn get_data(&self) -> Self::Data;
    fn get(&self, i: usize) -> f64;
}

pub trait ScalarSpace: Sized + LinearSpace {
    fn abs(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
}

pub trait InnerProduct: Sized + Mul<Self, Output = f64> {
    fn inner_product(&self, other: Self) -> f64;
}

pub trait VectorSpace: Sized + LinearSpace + InnerProduct + Index<usize> + IndexMut<usize> {
    fn magnitude(&self) -> f64;
    fn magnitude_square(&self) -> f64;
    fn normalize(&self) -> Self;
}

pub trait OuterProduct: VectorSpace {
    fn outer_product(&self, other: Self) -> Self;
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Scalar(f64);
#[derive(Clone, Copy, PartialEq)]
pub struct Vector2 {
    data: [f64; 2],
}

impl LinearSpace for Scalar {
    type Data = f64;

    fn new(data: Self::Data) -> Self {
        Self(data)
    }
    fn zero() -> Self {
        Self::new(0.0)
    }
    fn get_data(&self) -> Self::Data {
        self.0
    }
    fn get(&self, i: usize) -> f64 {
        match i {
            0 => self.0,
            _ => 0.0,
        }
    }
    fn size(&self) -> usize {
        1
    }
}

impl ScalarSpace for Scalar {
    fn abs(&self) -> Self {
        Self(self.0.abs())
    }
    fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }
    fn sin(&self) -> Self {
        Self(self.0.sin())
    }
    fn cos(&self) -> Self {
        Self(self.0.cos())
    }
}

impl LinearSpace for Vector2 {
    type Data = [f64; 2];

    fn new([e0, e1]: Self::Data) -> Self {
        Self { data: [e0, e1] }
    }
    fn zero() -> Self {
        Self::new([0.0, 0.0])
    }
    fn size(&self) -> usize {
        2
    }
    fn get_data(&self) -> Self::Data {
        self.data
    }
    fn get(&self, i: usize) -> f64 {
        self.data[i]
    }
}

impl VectorSpace for Vector2 {
    fn magnitude_square(&self) -> f64 {
        (self.data[0] * self.data[0]) + (self.data[1] * self.data[1])
    }
    fn magnitude(&self) -> f64 {
        self.magnitude_square().sqrt()
    }
    fn normalize(&self) -> Self {
        match self.magnitude() {
            0.0 => Self::zero(),
            _ => *self / self.magnitude(),
        }
    }
}

impl Index<usize> for Vector2 {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl InnerProduct for Vector2 {
    fn inner_product(&self, other: Self) -> f64 {
        (self.data[0] * other.data[0]) + (self.data[1] * other.data[1])
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scalar({:.6})", self.0)
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scalar({:.6})", self.0)
    }
}

impl fmt::Display for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector2({:.6}, {:.6})", self.data[0], self.data[1])
    }
}

impl fmt::Debug for Vector2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector2({:.6}, {:.6})", self.data[0], self.data[1])
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Neg for Vector2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            data: [-self.data[0], -self.data[1]],
        }
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: [self.data[0] + rhs.data[0], self.data[1] + rhs.data[1]],
        }
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Vector2) -> Self::Output {
        Self {
            data: [self.data[0] - rhs.data[0], self.data[1] - rhs.data[1]],
        }
    }
}

impl Mul<f64> for Scalar {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Mul<f64> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            data: [self.data[0] * rhs, self.data[1] * rhs],
        }
    }
}

impl Div<f64> for Scalar {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl Div<f64> for Vector2 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            data: [self.data[0] / rhs, self.data[1] / rhs],
        }
    }
}

impl Mul<Scalar> for f64 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Scalar(self * rhs.0)
    }
}

impl Mul<Vector2> for f64 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        Vector2 {
            data: [self * rhs.data[0], self * rhs.data[1]],
        }
    }
}

impl Mul<Self> for Vector2 {
    type Output = f64;

    fn mul(self, rhs: Self) -> Self::Output {
        self.inner_product(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn assert_f64_eq(a: f64, b: f64) {
        let diff = (a - b).abs();
        assert!(
            diff < EPS,
            "Assertion failed: {:?} != {:?} (diff: {})",
            a,
            b,
            diff
        );
    }

    fn assert_scalar_eq(a: Scalar, b: Scalar) {
        let diff = (a - b).abs();
        assert!(
            diff < Scalar::new(EPS),
            "Assertion failed: {:?} != {:?} (diff: {})",
            a,
            b,
            diff
        );
    }

    fn assert_vector_eq(a: Vector2, b: Vector2) {
        let diff_e0 = (a - b)[0].abs();
        let diff_e1 = (a - b)[1].abs();
        assert!(
            diff_e0 < EPS && diff_e1 < EPS,
            "Assertion failed: {:?} != {:?}",
            a,
            b
        )
    }

    #[test]
    fn test_scalar_op() {
        let s1 = Scalar::new(1.0);
        let s2 = Scalar::new(3.0);
        let f = 3.0;

        // Vector2 <op> Vector2
        assert_scalar_eq(s1 + s2, Scalar::new(4.0));
        assert_scalar_eq(s1 - s2, Scalar::new(-2.0));

        // Neg
        assert_scalar_eq(-s1, Scalar::new(-1.0));

        // Mul with Scalar and f64
        assert_scalar_eq(s1 * f, Scalar::new(3.0));
        assert_scalar_eq(f * s1, Scalar::new(3.0));

        // Div with Scalar and f64
        assert_scalar_eq(s1 / f, Scalar::new(1.0 / 3.0));
    }

    #[test]
    fn test_vector2_op() {
        let v1 = Vector2::new([1.0, 2.0]);
        let v2 = Vector2::new([3.0, 4.0]);
        let f = 3.0;

        // Vector2 <op> Vector2
        assert_vector_eq(v1 + v2, Vector2::new([4.0, 6.0]));
        assert_vector_eq(v1 - v2, Vector2::new([-2.0, -2.0]));
        // assert_vector_eq(v1 * v2, Vector2::new([3.0, 8.0]));

        // Neg
        assert_vector_eq(-v1, Vector2::new([-1.0, -2.0]));

        // Mul with Vector2 and f64
        assert_vector_eq(v1 * f, Vector2::new([3.0, 6.0]));
        assert_vector_eq(f * v1, Vector2::new([3.0, 6.0]));

        // Div with Vector2 and f64
        assert_vector_eq(v1 / f, Vector2::new([1.0 / 3.0, 2.0 / 3.0]));
    }

    #[test]
    fn test_vector_magnitude() {
        let v = Vector2::new([3.0, 4.0]);

        assert_f64_eq(v.magnitude(), 5.0);
        assert_f64_eq(v.magnitude_square(), 25.0);
    }

    #[test]
    fn test_vector_normalize() {
        let v1 = Vector2::new([3.0, 4.0]);
        let v2 = Vector2::zero();

        // Normalize
        let v1_normalized = v1.normalize();
        assert_f64_eq(v1_normalized.magnitude(), 1.0);
        assert_vector_eq(v1_normalized, Vector2::new([0.6, 0.8]));

        let v2_normalized = v2.normalize();
        assert_f64_eq(v2_normalized.magnitude(), 0.0);
        assert_vector_eq(v2_normalized, Vector2::new([0.0, 0.0]));
    }

    #[test]
    fn test_inner_product() {
        let v1 = Vector2::new([1.0, 0.0]);
        let v2 = Vector2::new([0.0, 1.0]);
        let v3 = Vector2::new([2.0, 2.0]);

        // Orthogonal
        assert_f64_eq(v1.inner_product(v2), 0.0);

        // Parallel
        assert_f64_eq(v1.inner_product(v3), 2.0);

        // Self inner product
        assert_f64_eq(v3.inner_product(v3), v3.magnitude_square());
    }
}
