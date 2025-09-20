use crypto_bigint::{U128, U384};
use crate::utils::{to_primitive, CrtContextTask1}; 
use rayon::prelude::*;

// Task 1B Strategy: Full CRT (5 primes) + M4R (16-bit blocks) + Parallelization.

const NUM_PRIMES: usize = 5;
const NUM_CHUNKS: usize = 8; // 32 vectors / 4 vectors/chunk
const TABLE_SIZE: usize = 1 << 16;

// Preprocessing for M4R. Generates tables for all 5 primes.
pub fn preprocess_m4r(a: &[U128; 32], primes: &[u64; NUM_PRIMES]) -> Vec<Vec<Vec<u64>>> {
    // Parallelize preprocessing over the primes.
    primes.par_iter().map(|&p| {
        let p_u128 = p as u128;
        // Reduce a_j mod p
        let a_mod_p: Vec<u64> = a.iter().map(|x| {
            (to_primitive(x) % p_u128) as u64
        }).collect();

        let mut tables = vec![vec![0u64; TABLE_SIZE]; NUM_CHUNKS];

        for chunk in 0..NUM_CHUNKS {
            let start = chunk * 4;
            // Pre-calculate the 4 coefficients (as u128 for multiplication).
            let a0 = a_mod_p[start] as u128;
            let a1 = a_mod_p[start + 1] as u128;
            let a2 = a_mod_p[start + 2] as u128;
            let a3 = a_mod_p[start + 3] as u128;

            for i in 0..TABLE_SIZE {
                let u0 = (i & 0xF) as u128;
                let u1 = ((i >> 4) & 0xF) as u128;
                let u2 = ((i >> 8) & 0xF) as u128;
                let u3 = ((i >> 12) & 0xF) as u128;

                // Compute sum = a0*u0 + ... + a3*u3 mod p
                let mut sum = a0 * u0;
                sum += a1 * u1;
                sum += a2 * u2;
                sum += a3 * u3;
                
                tables[chunk][i] = (sum % p_u128) as u64;
            }
        }
        tables
    }).collect()
}

// Main computation function for Task 1B.
pub fn task1b(
    b: &[U128],
    u_packed: &[u128],
    m: usize,
    n: usize,
    tables: &[Vec<Vec<u64>>], // [Prime][Chunk][Lookup]
    primes: &[u64; NUM_PRIMES],
    crt_context: &CrtContextTask1
) -> Vec<U384> {
    
    // 1. Pre-reduce b_i modulo all primes.
    let mut b_mod_p = vec![vec![0u64; m]; NUM_PRIMES];

    // Parallelize reduction: Iterate over the outer vector mutably.
    b_mod_p.par_iter_mut().enumerate().for_each(|(p_idx, b_mod_p_slice)| {
        let p = primes[p_idx];
        let p_u128 = p as u128;
        for (i, b_i) in b.iter().enumerate() {
            b_mod_p_slice[i] = (to_primitive(b_i) % p_u128) as u64;
        }
    });

    // 2. Compute residues for all N elements and all K primes.
    let mut residues = vec![[0u64; NUM_PRIMES]; n];

    // Parallelize over the N dimension (optimized for data locality).
    residues.par_iter_mut().enumerate().for_each(|(k, res_k)| {
        let u_k_slice = &u_packed[k*m..(k+1)*m];

        // Iterate over each prime sequentially within the thread.
        for p_idx in 0..NUM_PRIMES {
            let p_u128 = primes[p_idx] as u128;
            let tables_p = &tables[p_idx];
            let b_mod_p_p = &b_mod_p[p_idx];

            let mut sum_k = 0u64;

            for i in 0..m {
                let u_word = u_k_slice[i];

                // Inner sum using M4R (16-bit lookups). Accumulate in u128.
                let mut inner_sum_wide = 0u128;

                // Unroll the 8 lookups
                inner_sum_wide += tables_p[0][(u_word & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[1][((u_word >> 16) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[2][((u_word >> 32) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[3][((u_word >> 48) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[4][((u_word >> 64) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[5][((u_word >> 80) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[6][((u_word >> 96) & 0xFFFF) as usize] as u128;
                inner_sum_wide += tables_p[7][((u_word >> 112) & 0xFFFF) as usize] as u128;
                
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
    // Initialize using U384::ZERO.
    let mut v = vec![U384::ZERO; n];
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        *v_k = crt_context.reconstruct(&residues[k]);
    });

    v
}