use crate::math::{complex::{Complex, ComplexSpace}, core::{LinearSpace, Vector}};

pub fn dft1d<const N: usize>(x: Vector<Complex, N>) -> Vector<Complex, N> {
    Vector::zero()    // TODO: Implement
}

pub fn idft1d<const N: usize>(x: Vector<Complex, N>) -> Vector<Complex, N> {
    Vector::zero()    // TODO: Implement
}

pub fn fft1d<const N: usize>(x: Vector<Complex, N>) -> Vector<Complex, N> {
    Vector::zero()    // TODO: Implement
}

pub fn ifft1d<const N: usize>(x: Vector<Complex, N>) -> Vector<Complex, N> {
    Vector::zero()    // TODO: Implement
}

#[cfg(test)]
mod tests {
    use crate::math::core::ScalarSpace;

    use super::*;

    #[test]
    fn test_dft1d() {
        let x = Vector::new([Complex::zero(), Complex::one(), Complex::zero(), -Complex::one()]);
        let frequancy = Vector::new([Complex::zero(), Complex::new(0.0, -2.0), Complex::zero(), Complex::new(0.0, 2.0)]);
        let output = dft1d(x);
        assert_eq!(frequancy, output, "1D DFT")
    }

    #[test]
    fn test_idft1d() {
        let frequancy = Vector::new([Complex::zero(), Complex::new(0.0, -2.0), Complex::zero(), Complex::new(0.0, 2.0)]);
        let x = Vector::new([Complex::zero(), Complex::one(), Complex::zero(), -Complex::one()]);
        let output = idft1d(frequancy);
        assert_eq!(frequancy, output, "1D IDFT");
    }

    #[test]
    fn test_fft1d() {
        let x = Vector::new([Complex::zero(), Complex::one(), Complex::zero(), -Complex::one()]);
        let frequancy = Vector::new([Complex::zero(), Complex::new(0.0, -2.0), Complex::zero(), Complex::new(0.0, 2.0)]);
        let output = fft1d(x);
        assert_eq!(frequancy, output, "1D DFT")
    }

    #[test]
    fn test_ifft1d() {
        let frequancy = Vector::new([Complex::zero(), Complex::new(0.0, -2.0), Complex::zero(), Complex::new(0.0, 2.0)]);
        let x = Vector::new([Complex::zero(), Complex::one(), Complex::zero(), -Complex::one()]);
        let output = ifft1d(frequancy);
        assert_eq!(frequancy, output, "1D IDFT");
    }
}
