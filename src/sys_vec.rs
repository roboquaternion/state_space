/*!
# sys_vec
The `sys_vec` module contains `SysVec` struct, which is a convenient way to represent system
vectors, `u`, `x`, or `y`. Each vector has the same data type as `StateSpace` and the size
associated with `NU`, `NX`, or `NY`.

## Features.
* `SysVec` fields:
    * .val holds the vector.
    * .lb is the lower bound, default value is -9e99.
    * .ub is the upper bound, default value is +9e99.
* Several constructors, setters, and similar methods.
    * See the examples below.
*/


use na::SVector;
use nalgebra as na;

// A struct to hold a system vector and it's limits.
#[derive(Debug, Copy, Clone)]
pub struct SysVec<T, const N: usize> {
    val: SVector<T, N>,
    lb: SVector<T, N>,
    ub: SVector<T, N>,
}

// DEBUG, remove all println, replace with assert_eq or similar.

impl<T, const N: usize> SysVec<T, N>
where
    T: nalgebra::Scalar + PartialOrd + num_traits::NumCast,
{

    /// Construct a SysVec struct with default values: 0 for val and +/-9e99 for ub and lb.
    ///
    /// ```rust
    /// use nalgebra::{vector, SVector};
    /// use state_space::SysVec;
    /// type T = f32;
    /// const N: usize = 3;
    /// let my_vec: SysVec<T, N> = SysVec::new();   // my_vec.val is all zeros.
    ///
    /// let exp_val: SVector<T, N> = SVector::from_element(0.0);
    /// assert_eq!(exp_val, my_vec.get_val())
    /// ```
    pub fn new() -> Self {
        let zero = T::from(0).unwrap();
        let val_vec: SVector<T, N> = SVector::from_element(zero);

        let min_value: T = T::from(-9e99_f64).expect("Conversion failed");
        let lb_vec: SVector<T, N> = SVector::from_element(min_value);

        let max_value: T = T::from(9e99_f64).expect("Conversion failed");
        let ub_vec: SVector<T, N> = SVector::from_element(max_value);

        Self {
            val: val_vec,
            lb: lb_vec,
            ub: ub_vec,
        }
    }


    /// Construct a SysVec struct from a scalar value. All entries in the value matrix will be
    /// the same. The default values for lb and ub are applied.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f32;
    /// const N: usize = 3;
    /// let my_vec: SysVec<T, N> = SysVec::from_val(10.17);
    ///
    /// let exp_val: SVector<T, N> = SVector::from_element(10.17);
    /// assert_eq!(exp_val, my_vec.get_val());
    /// println!("my_vec = {:?}", my_vec);
    /// ```
    pub fn from_val(val: f64) -> Self {
        let val_value: T = T::from(val).expect("Conversion failed");
        let val_vec: SVector<T, N> = SVector::from_element(val_value);

        Self::new().set_val(val_vec).clone()
    }

    /// Construct a SysVec struct from scalar entries for val, lb, and ub. The scalar entries are
    /// repeated for all items in the corresponding matrices.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f32;
    /// const N: usize = 3;
    /// let my_vec: SysVec<T, N> = SysVec::from_scalars(10.17, -3.14, 6.28);
    ///
    /// let exp_val: SVector<T, N> = SVector::from_element(10.17);
    /// assert_eq!(exp_val, my_vec.get_val());
    ///
    /// let exp_lb: SVector<T, N> = SVector::from_element(-3.14);
    /// assert_eq!(exp_lb, my_vec.get_lb());
    ///
    /// let exp_ub: SVector<T, N> = SVector::from_element(6.28);
    /// assert_eq!(exp_ub, my_vec.get_ub());
    /// ```
    pub fn from_scalars(val: f64, lb: f64, ub: f64) -> Self {
        let val_value: T = T::from(val).expect("Conversion failed");
        let val_vec: SVector<T, N> = SVector::from_element(val_value);

        let min_value: T = T::from(lb).expect("Conversion failed");
        let lb_vec: SVector<T, N> = SVector::from_element(min_value);

        let max_value: T = T::from(ub).expect("Conversion failed");
        let ub_vec: SVector<T, N> = SVector::from_element(max_value);

        Self::new()
            .set_val(val_vec)
            .set_lb(lb_vec)
            .set_ub(ub_vec)
            .clone()
    }

    /// Setter for SysVec.val property. The input is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::new();
    ///
    /// let new_val: SVector<T, N> = SVector::from_element(0.1017);
    /// my_vec.set_val(new_val);
    ///
    /// let exp_val: SVector<T, N> = SVector::from_element(0.1017);
    /// assert_eq!(exp_val, my_vec.get_val());
    /// ```
    pub fn set_val(&mut self, val: SVector<T, N>) -> &mut SysVec<T, { N }> {
        self.val = val;
        self
    }

    /// Setter for SysVec.lb property. The input is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::new();
    ///
    /// let new_lb: SVector<T, N> = SVector::from_element(10.17);
    /// my_vec.set_lb(new_lb);
    ///
    /// let exp_lb: SVector<T, N> = SVector::from_element(10.17);
    /// assert_eq!(exp_lb, my_vec.get_lb());
    /// ```
    pub fn set_lb(&mut self, lb: SVector<T, N>) -> &mut Self {
        self.lb = lb;
        self
    }

    /// Setter for SysVec.ub property. The input is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::new();
    ///
    /// let new_ub: SVector<T, N> = SVector::from_element(10.17);
    /// my_vec.set_ub(new_ub);
    ///
    /// let exp_ub: SVector<T, N> = SVector::from_element(10.17);
    /// assert_eq!(exp_ub, my_vec.get_ub());
    /// ```
    pub fn set_ub(&mut self, ub: SVector<T, N>) -> &mut Self {
        self.ub = ub;
        self
    }

    /// Getter for SysVec.val property. The output is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::from_scalars(1.017, -3.14, 6.28);
    ///
    /// let exp_val: SVector<T, N> = SVector::from_element(1.017);
    /// assert_eq!(exp_val, my_vec.get_val());
    /// ```
    pub fn get_val(&self) -> SVector<T, N> {
        self.val.clone()
    }

    /// Getter for SysVec.lb property. The output is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::from_scalars(1.017, -3.14, 6.28);
    ///
    /// let exp_lb: SVector<T, N> = SVector::from_element(-3.14);
    /// assert_eq!(exp_lb, my_vec.get_lb());
    /// ```
    pub fn get_lb(&self) -> SVector<T, N> {
        self.lb.clone()
    }

    /// Getter for SysVec.ub property. The output is an SVector.
    ///
    /// ```rust
    /// use nalgebra::SVector;
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// let mut my_vec: SysVec<T, N> = SysVec::from_scalars(1.017, -3.14, 6.28);
    ///
    /// let exp_ub: SVector<T, N> = SVector::from_element(6.28);
    /// assert_eq!(exp_ub, my_vec.get_ub());
    /// ```
    pub fn get_ub(&self) -> SVector<T, N> {
        self.ub.clone()
    }

    /// Use the clamp() method to gurantee that all self.lb <= self.val <= self.ub.
    /// ///
    /// ```rust
    /// use nalgebra::{SVector, vector};
    /// use state_space::SysVec;
    /// type T = f64;
    /// const N: usize = 3;
    ///
    /// // Make a SysVec struct.
    /// let val = vector![1.0, -314.1, 628.3];
    /// let lb = vector![-5.0, -5.0, -5.0];
    /// let ub = vector![9.0, 9.0, 9.0];
    /// let mut my_vec: SysVec<T, N> = *SysVec::new()
    ///     .set_val(val)
    ///     .set_lb(lb)
    ///     .set_ub(ub);
    ///
    /// // Apply the clamp.
    /// my_vec.clamp();
    ///
    /// let exp_val: SVector<T, N> = vector![1.0, -5.0, 9.0];
    /// assert_eq!(exp_val, my_vec.get_val())
    /// ```
    pub fn clamp(&mut self) -> &mut Self {
        self.val = self
            .val
            .zip_zip_map(&self.lb, &self.ub, |x, min, max| na::clamp(x, min, max));
        self
    }

    /// This method is used in StateSpace.update(). It updates the val property and checks clamp.
    /// End users do not need to be concerned with this method.
    pub fn update(&mut self, val: SVector<T, N>) -> &mut Self {
        self.val = val;
        self.clamp();
        self
    }
}

impl<T, const N: usize> Default for SysVec<T, N>
where
    T: nalgebra::Scalar + PartialOrd + num_traits::NumCast,
 {
    fn default() -> Self {
        Self::new()
    }
}
