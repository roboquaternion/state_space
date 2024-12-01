use na::SMatrix;
use nalgebra as na;
use state_space::{StateSpace, SysVec};

fn main() {
    type T = f64;
    const NU: usize = 1;
    const NX: usize = 1;
    const NY: usize = 1;

    // a matrix from a vector.
    let v: Vec<T> = (1..=NX * NX).map(|x| x as T).collect(); // DEBUG, get a ^2 or ** or pow.
    let _a: SMatrix<T, NX, NX> = SMatrix::from_vec(v);

    // Create a SysVec.
    let svu: SysVec<T, NU> = SysVec::from_scalars(1.0, -9e99, 9e99);
    let x0: SysVec<T, NX> = SysVec::from_val(0.1);

    // Create a state space system.
    let mut sys: StateSpace<T, NU, NX, NY> = StateSpace::new();
    sys.set_a(-1.000 * SMatrix::<T, NX, NX>::identity())
        .set_b( 1.000 * SMatrix::<T, NX, NU>::identity())
        .set_c( 1.000 * SMatrix::<T, NY, NX>::identity())
        .set_x(x0)
        .set_dt(0.1);

    // Create a step input.
    sys.set_u(svu);

    // Update many times.
    for _ in 0..50 {
        sys.update();
        println!("sys.y.val = {:?}", sys.get_y());
    }
}
