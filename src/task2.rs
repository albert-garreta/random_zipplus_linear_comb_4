use crypto_bigint::{U128, U192};
use crate::utils::{to_primitive, CrtContextTask2}; 
use rayon::prelude::*;

// Task 2 Strategy: Full CRT (3 primes) + Parallelization.

const NUM_PRIMES: usize = 3;

// Main computation function for Task 2.
pub fn task2(
    b: &[U128],
    u: &[Vec<u32>],
    m: usize,
    n: usize,
    primes: &[u64; NUM_PRIMES],
    crt_context: &CrtContextTask2
) -> Vec<U192> {

    // 1. Pre-reduce b_i modulo all primes.
    // Structure: b_mod_p[Prime_idx][i]
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
        
        // Iterate over each prime sequentially within the thread.
        for p_idx in 0..NUM_PRIMES {
            let p = primes[p_idx];
            let b_mod_p_p = &b_mod_p[p_idx];

            // Accumulate in u128 to minimize modulo operations.
            let mut sum_k = 0u128; 
            
            for i in 0..m {
                // b_i (64-bit) * u_i[k] (32-bit). Product is 96-bit.
                let product = b_mod_p_p[i] as u128 * u[i][k] as u128;
                sum_k += product;
            }
            // Single modulo reduction at the end.
            res_k[p_idx] = (sum_k % p as u128) as u64;
        }
    });

    // 3. CRT Reconstruction (Parallelized over N).
    // Initialize using U192::ZERO.
    let mut v = vec![U192::ZERO; n];
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        *v_k = crt_context.reconstruct(&residues[k]);
    });

    v
}