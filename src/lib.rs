// Entry point for this project.

// DEBUG: Items to add:
// 1. reset(), of course this is just set_x()...
// 2. enum for continuous, forward-euler, ...
// 3. Documentation.

pub mod sys_vec; // re-export.

use na::SMatrix;
use nalgebra as na;
use num_traits::{NumCast, One, Zero};
use sys_vec::SysVec;

#[derive(Debug, Copy, Clone)]
pub struct StateSpace<T, const NU: usize, const NX: usize, const NY: usize> {
    a: SMatrix<T, NX, NX>,
    b: SMatrix<T, NX, NU>,
    c: SMatrix<T, NY, NX>,
    d: SMatrix<T, NY, NU>,
    u: SysVec<T, NU>,
    x: SysVec<T, NX>,
    y: SysVec<T, NY>,
    pub dt: T,
}

impl<T, const NU: usize, const NX: usize, const NY: usize> StateSpace<T, NU, NX, NY>
where
    T: nalgebra::Scalar
        + nalgebra::ClosedAddAssign
        + nalgebra::ClosedMulAssign
        + PartialOrd
        + One
        + Zero
        + NumCast,
{
    pub fn new() -> Self {
        // System matrices.
        let a: SMatrix<T, NX, NX> = SMatrix::from_element(Zero::zero());
        let b: SMatrix<T, NX, NU> = SMatrix::from_element(Zero::zero());
        let c: SMatrix<T, NY, NX> = SMatrix::from_element(Zero::zero());
        let d: SMatrix<T, NY, NU> = SMatrix::from_element(Zero::zero());

        // System vectors, with bounds.
        let u: SysVec<T, NU> = SysVec::new();
        let x: SysVec<T, NX> = SysVec::new();
        let y: SysVec<T, NY> = SysVec::new();

        // Create the new struct and return it.
        Self {
            a,
            b,
            c,
            d,
            u,
            x,
            y,
            dt: T::one(),
        }
    }

    pub fn set_a(&mut self, mat: SMatrix<T, NX, NX>) -> &mut Self {
        self.a = mat;
        self
    }

    pub fn set_b(&mut self, mat: SMatrix<T, NX, NU>) -> &mut Self {
        self.b = mat;
        self
    }

    pub fn set_c(&mut self, mat: SMatrix<T, NY, NX>) -> &mut Self {
        self.c = mat;
        self
    }

    pub fn set_d(&mut self, mat: SMatrix<T, NY, NU>) -> &mut Self {
        self.d = mat;
        self
    }

    pub fn set_u(&mut self, vec: SysVec<T, NU>) -> &mut Self {
        self.u = vec;
        self
    }

    pub fn set_x(&mut self, vec: SysVec<T, NX>) -> &mut Self {
        self.x = vec;
        self
    }

    pub fn set_y(&mut self, vec: SysVec<T, NY>) -> &mut Self {
        self.y = vec;
        self
    }

    pub fn set_dt(&mut self, dt: T) -> &mut Self {
        self.dt = dt;
        self
    }

    pub fn get_u(&self) -> SMatrix<T, NU, 1> {
        self.u.get_val()
    }

    pub fn get_x(&self) -> SMatrix<T, NX, 1> {
        self.x.get_val()
    }

    pub fn get_y(&self) -> SMatrix<T, NY, 1> {
        self.y.get_val()
    }

    // Apply forward-euler equations to move forward in time by dt time units.
    // This is the continuous time version of the equation.
    pub fn update(&mut self) -> &mut Self {
        // Check u and x for clamp, update self.
        self.u.clamp();
        self.x.clamp();

        // Local variables for x(n) and u(n).
        let u0 = self.u.get_val();
        let x0 = self.x.get_val();

        // Derivative equation. xDot = Ax + Bu.
        let x_dot: SMatrix<T, NX, 1> =
            (self.a.clone() * x0.clone()) + (self.b.clone() * u0.clone());

        // This is a super simple integrator, Forward Euler. Also known as x(n+1).
        let x1 = x0.clone() + x_dot * self.dt.clone();
        self.x = self.x.clone().update(x1).to_owned();

        // Output equation, y = Cx + Du. It uses x(n), not x(n+1), for forward euler technique.
        let yn = (self.c.clone() * x0.clone()) + (self.d.clone() * u0.clone());
        self.y = self.y.clone().update(yn).to_owned();

        self
    }
}

impl<T, const NU: usize, const NX: usize, const NY: usize> Default for StateSpace<T, NU, NX, NY>
where
    T: nalgebra::Scalar
        + nalgebra::ClosedAddAssign
        + nalgebra::ClosedMulAssign
        + PartialOrd
        + One
        + Zero
        + NumCast,
 {
    fn default() -> Self {
        Self::new()
    }
}
