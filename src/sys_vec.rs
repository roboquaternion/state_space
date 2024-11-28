// This module contains the SysVec structure and functions that are implemented
// on it.

use na::SMatrix;
use nalgebra as na;

// A struct to hold a system vector and it's limits.
#[derive(Debug, Copy, Clone)]
pub struct SysVec<T, const N: usize> {
    val: SMatrix<T, N, 1>,
    lb: SMatrix<T, N, 1>,
    ub: SMatrix<T, N, 1>,
}

impl<T, const N: usize> SysVec<T, N>
where
    T: nalgebra::Scalar + core::cmp::PartialOrd + num_traits::NumCast,
{
    // A constructor like method that sets all the default values (0, -9e99, and +9e99).
    pub fn new() -> Self {
        let zero = T::from(0).unwrap();
        let val_vec: SMatrix<T, N, 1> = SMatrix::from_element(zero);

        let min_value: T = T::from(-9e99_f64).expect("Conversion failed");
        let lb_vec: SMatrix<T, N, 1> = SMatrix::from_element(min_value);

        let max_value: T = T::from(9e99_f64).expect("Conversion failed");
        let ub_vec: SMatrix<T, N, 1> = SMatrix::from_element(max_value);

        SysVec {
            val: val_vec,
            lb: lb_vec,
            ub: ub_vec,
        }
    }

    // A constructor-like method that takes a scalar for val and uses defaults
    // for lb and ub.
    pub fn from_val(val: f64) -> Self {
        let val_value: T = T::from(val).expect("Conversion failed");
        let val_vec: SMatrix<T, N, 1> = SMatrix::from_element(val_value);
        Self::new().set_val(val_vec)
    }

    // A constructor like method that takes a scalar value for val, lb, and ub and
    // creates a SysVec of the correct size.
    pub fn from_scalars(val: f64, lb: f64, ub: f64) -> Self {
        let val_value: T = T::from(val).expect("Conversion failed");
        let val_vec: SMatrix<T, N, 1> = SMatrix::from_element(val_value);

        let min_value: T = T::from(lb).expect("Conversion failed");
        let lb_vec: SMatrix<T, N, 1> = SMatrix::from_element(min_value);

        let max_value: T = T::from(ub).expect("Conversion failed");
        let ub_vec: SMatrix<T, N, 1> = SMatrix::from_element(max_value);

        Self::new().set_val(val_vec).set_lb(lb_vec).set_ub(ub_vec)
    }

    pub fn set_val(mut self, val: SMatrix<T, N, 1>) -> Self {
        self.val = val;
        self
    }

    pub fn set_lb(mut self, lb: SMatrix<T, N, 1>) -> Self {
        self.lb = lb;
        self
    }

    pub fn set_ub(mut self, ub: SMatrix<T, N, 1>) -> Self {
        self.ub = ub;
        self
    }

    pub fn get_val(&self) -> SMatrix<T, N, 1> {
        self.val.clone()
    }

    pub fn clamp(&mut self) -> &mut Self {
        self.val = self
            .val
            .zip_zip_map(&self.lb, &self.ub, |x, min, max| na::clamp(x, min, max));
        self
    }

    pub fn update(&mut self, val: SMatrix<T, N, 1>) -> &mut Self {
        self.val = val;
        self.clamp();
        self
    }
}

impl<T, const N: usize> Default for SysVec<T, N>
where
    T: nalgebra::Scalar + core::cmp::PartialOrd + num_traits::NumCast,
 {
    fn default() -> Self {
        Self::new()
    }
}
