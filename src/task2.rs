use crypto_bigint::U128;
use crate::utils::{P64, to_primitive};
use rayon::prelude::*;

// Task 2 Strategy: CRT (mod P64) + Parallelization.

// Main computation function for Task 2.
pub fn task2(b: &[U128], u: &[Vec<u32>], m: usize, n: usize) -> Vec<u64> {
    // Reduce b_i mod P64
    let b_mod_p: Vec<u64> = b.iter().map(|x| {
        (to_primitive(x) % P64 as u128) as u64
    }).collect();

    let mut v = vec![0u64; n];

    // Parallelize over the n dimension
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        // Optimization: Accumulate in u128 to minimize modulo operations.
        // Products are 96-bit (64*32). If m < 2^32, the sum fits in u128.
        let mut sum_k = 0u128; 
        
        for i in 0..m {
            // Note: Accessing u[i][k] might not be cache-optimal if u is row-major.
            let product = b_mod_p[i] as u128 * u[i][k] as u128;
            sum_k += product;
        }
        // Single modulo reduction at the end.
        *v_k = (sum_k % P64 as u128) as u64;
    });

    v
}