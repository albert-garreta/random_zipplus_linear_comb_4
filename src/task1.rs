use crypto_bigint::U128;
use crate::utils::{P64, to_primitive};
use rayon::prelude::*;

// Task 1 Strategy: CRT (mod P64) + M4R (8-bit blocks) + Parallelization.

// Preprocessing for M4R (Excluded from benchmark time).
pub fn preprocess_m4r(a: &[U128; 32]) -> Vec<Vec<u64>> {
    // Reduce a_j mod P64
    let a_mod_p: Vec<u64> = a.iter().map(|x| {
        (to_primitive(x) % P64 as u128) as u64
    }).collect();

    // 4 tables of size 256 (8-bit blocks)
    let mut tables = vec![vec![0u64; 256]; 4];

    for chunk in 0..4 {
        let start = chunk * 8;
        for i in 0..256 {
            let mut sum = 0u64;
            for j in 0..8 {
                if (i >> j) & 1 == 1 {
                    // Modular addition
                    sum = (sum + a_mod_p[start + j]) % P64;
                }
            }
            tables[chunk][i] = sum;
        }
    }
    tables
}

// Main computation function for Task 1.
pub fn task1(b: &[U128], u_transposed: &[u32], m: usize, n: usize, tables: &[Vec<u64>]) -> Vec<u64> {
    // Reduce b_i mod P64
    let b_mod_p: Vec<u64> = b.iter().map(|x| {
        (to_primitive(x) % P64 as u128) as u64
    }).collect();

    let mut v = vec![0u64; n];

    // Parallelize over the n dimension
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        let mut sum_k = 0u64;
        
        // Access the transposed data (cache-friendly)
        let u_k_slice = &u_transposed[k*m..(k+1)*m];

        for i in 0..m {
            let u_word = u_k_slice[i];

            // Compute the inner sum using M4R lookup tables.
            // Optimization: Accumulate the 4 lookups in u128 to reduce modulo operations.
            let mut inner_sum_wide = 0u128;
            inner_sum_wide += tables[0][(u_word & 0xFF) as usize] as u128;
            inner_sum_wide += tables[1][((u_word >> 8) & 0xFF) as usize] as u128;
            inner_sum_wide += tables[2][((u_word >> 16) & 0xFF) as usize] as u128;
            inner_sum_wide += tables[3][((u_word >> 24) & 0xFF) as usize] as u128;
            
            let inner_sum = (inner_sum_wide % P64 as u128) as u64;

            // Outer sum accumulation: (b_i * inner_sum) mod P64
            let product = (b_mod_p[i] as u128 * inner_sum as u128) % P64 as u128;
            sum_k = (sum_k + product as u64) % P64;
        }
        *v_k = sum_k;
    });

    v
}