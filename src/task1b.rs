use crypto_bigint::U128;
use crate::utils::{P64, to_primitive};
use rayon::prelude::*;

// Task 1B Strategy: CRT (mod P64) + M4R (16-bit blocks) + Parallelization.

// Preprocessing for M4R (Excluded from benchmark time).
// Assumes entries are 4-bit (N <= 16384).
pub fn preprocess_m4r(a: &[U128; 32]) -> Vec<Vec<u64>> {
    // Reduce a_j mod P64
    let a_mod_p: Vec<u64> = a.iter().map(|x| {
        (to_primitive(x) % P64 as u128) as u64
    }).collect();

    // 8 tables of size 65536 (16-bit blocks = 4 vectors * 4 bits/entry)
    const TABLE_SIZE: usize = 1 << 16;
    let mut tables = vec![vec![0u64; TABLE_SIZE]; 8];

    for chunk in 0..8 {
        let start = chunk * 4;
        // Pre-calculate the 4 coefficients for this chunk
        let a0 = a_mod_p[start] as u128;
        let a1 = a_mod_p[start + 1] as u128;
        let a2 = a_mod_p[start + 2] as u128;
        let a3 = a_mod_p[start + 3] as u128;

        for i in 0..TABLE_SIZE {
            // i represents the packed 16 bits (u0, u1, u2, u3)
            // u0 is the least significant 4 bits.
            let u0 = (i & 0xF) as u128;
            let u1 = ((i >> 4) & 0xF) as u128;
            let u2 = ((i >> 8) & 0xF) as u128;
            let u3 = ((i >> 12) & 0xF) as u128;

            // Compute sum = a0*u0 + a1*u1 + a2*u2 + a3*u3 mod P64
            // Use u128 for intermediate products.
            let mut sum = 0u128;
            sum += a0 * u0;
            sum += a1 * u1;
            sum += a2 * u2;
            sum += a3 * u3;
            
            tables[chunk][i] = (sum % P64 as u128) as u64;
        }
    }
    tables
}

// Main computation function for Task 1B.
// u_packed contains packed u128 words.
pub fn task1b(b: &[U128], u_packed: &[u128], m: usize, n: usize, tables: &[Vec<u64>]) -> Vec<u64> {
    // Reduce b_i mod P64
    let b_mod_p: Vec<u64> = b.iter().map(|x| {
        (to_primitive(x) % P64 as u128) as u64
    }).collect();

    let mut v = vec![0u64; n];

    // Parallelize over the n dimension
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        let mut sum_k = 0u64;
        
        // Access the transposed data (cache-friendly)
        let u_k_slice = &u_packed[k*m..(k+1)*m];

        for i in 0..m {
            let u_word = u_k_slice[i];

            // Compute the inner sum using M4R lookup tables.
            // Optimization: Accumulate the 8 lookups in u128 to reduce modulo operations.
            let mut inner_sum_wide = 0u128;

            // Unroll the 8 lookups from the 128-bit word (16 bits per lookup)
            // The compiler optimizes these shifts and masks efficiently.
            inner_sum_wide += tables[0][(u_word & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[1][((u_word >> 16) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[2][((u_word >> 32) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[3][((u_word >> 48) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[4][((u_word >> 64) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[5][((u_word >> 80) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[6][((u_word >> 96) & 0xFFFF) as usize] as u128;
            inner_sum_wide += tables[7][((u_word >> 112) & 0xFFFF) as usize] as u128;
            
            let inner_sum = (inner_sum_wide % P64 as u128) as u64;

            // Outer sum accumulation: (b_i * inner_sum) mod P64
            let product = (b_mod_p[i] as u128 * inner_sum as u128) % P64 as u128;
            sum_k = (sum_k + product as u64) % P64;
        }
        *v_k = sum_k;
    });

    v
}