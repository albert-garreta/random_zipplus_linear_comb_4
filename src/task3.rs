// Import paths compatible with crypto-bigint v0.5.5.
use crypto_bigint::{U128, modular::runtime_mod::{DynResidueParams, DynResidue}};
use std::sync::Arc;
use rayon::prelude::*;
use std::ptr;

// Task 3 Strategy: Montgomery Arithmetic + Efficient mu_i + M4R + Parallelization.

const LIMBS: usize = U128::LIMBS;
// Use Arc to share parameters across threads safely (Sync).
type Params = Arc<DynResidueParams<LIMBS>>;
type Residue = DynResidue<LIMBS>;

// --- Unsafe Workaround for crypto-bigint v0.5.5 ---
// In v0.5.5, DynResidue does not implement Clone. 
// We use unsafe pointer reads to manually copy these structures for performance.

// SAFETY: DynResidue is composed of Uint and DynResidueParams (POD-like structures of integers). 
// It does not manage heap resources or have custom Drop behavior. A bitwise copy is safe.
unsafe fn clone_residue<const L: usize>(residue: &DynResidue<L>) -> DynResidue<L> {
    ptr::read(residue)
}
// ----------------------------------------------------


// Helper function to compute mu_i values efficiently (O(m) time).
fn compute_mu(b_t: &[U128], t: usize, params: Params) -> Vec<Residue> {
    let m = 1 << t;
    let mut mu = Vec::with_capacity(m);

    // DynResidue::new consumes params. We dereference Arc and use the safe Clone implementation of DynResidueParams.
    let b_monty: Vec<_> = b_t.iter().map(|bi| {
        let params_copy = (*params).clone();
        DynResidue::new(bi, params_copy)
    }).collect();

    // mu[0] = 1
    let params_copy_one = (*params).clone();
    mu.push(DynResidue::one(params_copy_one));

    // Generate mu_i iteratively (Gray code sequence generation)
    for i in 0..t {
        let len = mu.len();
        let b_i = &b_monty[i];
        for j in 0..len {
            let product = mu[j].mul(b_i);
            mu.push(product);
        }
    }
    mu
}

// Preprocessing for M4R (Excluded from benchmark time).
pub fn preprocess_m4r_montgomery(a: &[U128; 32], params: Params) -> Vec<Vec<Residue>> {
    let a_monty: Vec<_> = a.iter().map(|ai| {
        let params_copy = (*params).clone();
        DynResidue::new(ai, params_copy)
    }).collect();

    let mut tables = Vec::with_capacity(4);
    for chunk in 0..4 {
        let start = chunk * 8;
        let mut table = Vec::with_capacity(256);
        for i in 0..256 {
            let params_copy_zero = (*params).clone();
            let mut sum = DynResidue::zero(params_copy_zero);
            
            for j in 0..8 {
                if (i >> j) & 1 == 1 {
                    sum = sum.add(&a_monty[start + j]);
                }
            }
            table.push(sum);
        }
        tables.push(table);
    }
    tables
}

// Main computation function for Task 3.
pub fn task3(
    q: U128,
    b_t: &[U128],
    u_transposed: &[u32],
    m: usize,
    n: usize,
    tables: &[Vec<Residue>]
) -> Vec<U128> {
    let t = m.ilog2() as usize;

    // 1. Montgomery Setup (Included in benchmark)
    let params = Arc::new(DynResidueParams::new(&q));

    // 2. Compute mu_i values (Included in benchmark)
    let mu = compute_mu(b_t, t, params.clone());

    // 3. Main computation

    // Initialize vector v. Since Residue is not Clone, we must manually initialize.
    // Create a template zero residue first.
    let params_copy_zero = (*params).clone();
    let zero_residue = DynResidue::zero(params_copy_zero);
    
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // SAFETY: Use unsafe clone for residue initialization.
        v.push(unsafe { clone_residue(&zero_residue) });
    }

    // Parallelize over the n dimension
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        
        let u_k_slice = &u_transposed[k*m..(k+1)*m];

        if m == 0 { return; }

        // Optimization: Initialize the accumulator (sum_k) starting from the first element (i=0), 
        // rather than initializing to zero. This avoids creating/cloning a zero residue in the loop.

        // --- First iteration (i=0) to initialize sum_k ---
        let u_word_0 = u_k_slice[0];
        
        // Initialize inner_sum by copying the first lookup result.
        // SAFETY: Unsafe clone of the residue from the table.
        let mut inner_sum_0 = unsafe { clone_residue(&tables[0][(u_word_0 & 0xFF) as usize]) };
        inner_sum_0 = inner_sum_0.add(&tables[1][((u_word_0 >> 8) & 0xFF) as usize]);
        inner_sum_0 = inner_sum_0.add(&tables[2][((u_word_0 >> 16) & 0xFF) as usize]);
        inner_sum_0 = inner_sum_0.add(&tables[3][((u_word_0 >> 24) & 0xFF) as usize]);

        // Initialize sum_k
        let mut sum_k = mu[0].mul(&inner_sum_0);

        // --- Remaining iterations (i=1..m) ---
        for i in 1..m {
            let u_word = u_k_slice[i];

            // Initialize inner_sum by copying the first lookup result.
            // SAFETY: Unsafe clone.
            let mut inner_sum = unsafe { clone_residue(&tables[0][(u_word & 0xFF) as usize]) };
            inner_sum = inner_sum.add(&tables[1][((u_word >> 8) & 0xFF) as usize]);
            inner_sum = inner_sum.add(&tables[2][((u_word >> 16) & 0xFF) as usize]);
            inner_sum = inner_sum.add(&tables[3][((u_word >> 24) & 0xFF) as usize]);

            // Outer sum accumulation: mu_i * inner_sum
            let product = mu[i].mul(&inner_sum);
            sum_k = sum_k.add(&product);
        }
        *v_k = sum_k;
    });

    // 4. Convert result back from Montgomery form
    v.into_iter().map(|vk| vk.retrieve()).collect()
}