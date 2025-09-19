// Import paths compatible with crypto-bigint v0.5.5.
use crypto_bigint::{U128, modular::runtime_mod::{DynResidueParams, DynResidue}};
use std::sync::Arc;
use rayon::prelude::*;
use std::ptr;

// Task 4 Strategy: Montgomery Arithmetic + Efficient mu_i + Parallelization.

const LIMBS: usize = U128::LIMBS;
type Params = Arc<DynResidueParams<LIMBS>>;
type Residue = DynResidue<LIMBS>;

// --- Unsafe Workaround for crypto-bigint v0.5.5 ---
// (See task3.rs for detailed explanation)

// SAFETY: DynResidue is POD-like.
unsafe fn clone_residue<const L: usize>(residue: &DynResidue<L>) -> DynResidue<L> {
    ptr::read(residue)
}
// ----------------------------------------------------

// Reusing compute_mu logic (O(m) time)
fn compute_mu(b_t: &[U128], t: usize, params: Params) -> Vec<Residue> {
    let m = 1 << t;
    let mut mu = Vec::with_capacity(m);

    // DynResidue::new consumes params. We dereference Arc and use the safe Clone.
    let b_monty: Vec<_> = b_t.iter().map(|bi| {
        let params_copy = (*params).clone();
        DynResidue::new(bi, params_copy)
    }).collect();

    // mu[0] = 1
    let params_copy_one = (*params).clone();
    mu.push(DynResidue::one(params_copy_one));

    // Generate mu_i iteratively
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

// Main computation function for Task 4.
pub fn task4(
    q: U128,
    b_t: &[U128],
    u: &[Vec<u32>],
    m: usize,
    n: usize
) -> Vec<U128> {
    let t = m.ilog2() as usize;

    // 1. Montgomery Setup (Included in benchmark)
    let params = Arc::new(DynResidueParams::new(&q));

    // 2. Compute mu_i values (Included in benchmark)
    let mu = compute_mu(b_t, t, params.clone());

    // 3. Main computation

    // Create a zero residue template.
    let params_copy_zero = (*params).clone();
    let zero_residue = DynResidue::zero(params_copy_zero);

    // Initialize vector v using unsafe clone.
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // SAFETY: Use unsafe clone for residue initialization.
        v.push(unsafe { clone_residue(&zero_residue) });
    }

    // Parallelize over the n dimension
    v.par_iter_mut().enumerate().for_each(|(k, v_k)| {
        // Initialize local accumulator sum_k.
        // SAFETY: Use unsafe clone of the zero residue template.
        let mut sum_k = unsafe { clone_residue(&zero_residue) };

        for i in 0..m {
            // Compute mu[i] * u[i][k] mod q.

            // Convert u[i][k] (32-bit) to Montgomery form.
            let u_val = U128::from_u32(u[i][k]);
            
            // We need a copy of the parameters for this conversion. Use safe clone.
            // This clone happens O(N*M) times.
            let params_copy = (*params).clone();
            let u_monty = DynResidue::new(&u_val, params_copy);
            
            // Montgomery multiplication and addition
            let product = mu[i].mul(&u_monty);
            sum_k = sum_k.add(&product);
        }
        *v_k = sum_k;
    });

    // 4. Convert result back from Montgomery form
    v.into_iter().map(|vk| vk.retrieve()).collect()
}