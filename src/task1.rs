use crypto_bigint::{U128, U384};
use crate::utils::{to_primitive, CrtContextTask1};
use rayon::prelude::*;

// Task 1 Strategy: Full CRT (5 primes) + M4R (8-bit blocks) + Parallelization.

const NUM_PRIMES: usize = 5;
const NUM_CHUNKS: usize = 4; // 32 bits / 8 bits/chunk

// Preprocessing for M4R. Generates tables for all 5 primes.
// Output: [Prime_idx][Chunk_idx][Lookup_val]
pub fn preprocess_m4r(a: &[U128; 32], primes: &[u64; NUM_PRIMES]) -> Vec<Vec<Vec<u64>>> {
    // Parallelize preprocessing over the primes.
    primes.par_iter().map(|&p| {
        let p_u128 = p as u128;
        // Reduce a_j mod p
        let a_mod_p: Vec<u64> = a.iter().map(|x| {
            (to_primitive(x) % p_u128) as u64
        }).collect();

        let mut tables = vec![vec![0u64; 256]; NUM_CHUNKS];
        for chunk in 0..NUM_CHUNKS {
            let start = chunk * 8;
            for i in 0..256 {
                let mut sum = 0u64;
                for j in 0..8 {
                    if (i >> j) & 1 == 1 {
                        // Modular addition. Use u128 for safety with large 64-bit primes.
                        sum = ((sum as u128 + a_mod_p[start + j] as u128) % p_u128) as u64;
                    }
                }
                tables[chunk][i] = sum;
            }
        }
        tables
    }).collect()
}

// Main computation function for Task 1.
pub fn task1(
    b: &[U128],
    u_transposed: &[u32],
    m: usize,
    n: usize,
    tables: &[Vec<Vec<u64>>], // [Prime][Chunk][Lookup]
    primes: &[u64; NUM_PRIMES],
    crt_context: &CrtContextTask1
) -> Vec<U384> {
    
    // 1. Pre-reduce b_i modulo all primes.
    // Structure: b_mod_p[Prime_idx][i]
    let mut b_mod_p = vec![vec![0u64; m]; NUM_PRIMES];
    
    // Parallelize reduction: Iterate over the outer vector mutably.
    b_mod_p.par_iter_mut().enumerate().for_each(|(p_idx, b_mod_p_slice)| {
        let p = primes[p_idx];
        let p_u128 = p as u128;
        for (i, b_i) in b.iter().enumerate() {
            // b_mod_p_slice is the mutable slice for the current prime.
            b_mod_p_slice[i] = (to_primitive(b_i) % p_u128) as u64;
        }
    });

    // 2. Compute residues for all N elements and all K primes.
    let mut residues = vec![[0u64; NUM_PRIMES]; n];

    // Parallelize over the N dimension (optimized for data locality).
    residues.par_iter_mut().enumerate().for_each(|(k, res_k)| {
        let u_k_slice = &u_transposed[k*m..(k+1)*m];

        // Iterate over each prime sequentially within the thread.
        for p_idx in 0..NUM_PRIMES {
            let p_u128 = primes[p_idx] as u128;
            let tables_p = &tables[p_idx];
            let b_mod_p_p = &b_mod_p[p_idx];

            let mut sum_k = 0u64;

            for i in 0..m {
                let u_word = u_k_slice[i];

                // Inner sum using M4R. Accumulate in u128.
                let mut inner_sum_wide = 0u128;
                inner_sum_wide += tables_p[0][(u_word & 0xFF) as usize] as u128;
                inner_sum_wide += tables_p[1][((u_word >> 8) & 0xFF) as usize] as u128;
                inner_sum_wide += tables_p[2][((u_word >> 16) & 0xFF) as usize] as u128;
                inner_sum_wide += tables_p[3][((u_word >> 24) & 0xFF) as usize] as u128;
                
                let inner_sum = (inner_sum_wide % p_u128) as u64;

                // Outer sum accumulation: (b_i * inner_sum) mod p
                let product = (b_mod_p_p[i] as u128 * inner_sum as u128) % p_u128;
                // Safe modular addition.
                sum_k = ((sum_k as u128 + product) % p_u128) as u64;
            }
            res_k[p_idx] = sum_k;
        }
    });

    // 3. CRT Reconstruction (Parallelized over N).
    // Initialize using U384::ZERO (associated constant).
    let mut v = vec![U384::ZERO; n];
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        *v_k = crt_context.reconstruct(&residues[k]);
    });

    v
}