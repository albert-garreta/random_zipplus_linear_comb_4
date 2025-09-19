https://g.co/gemini/share/60982577e656

In benchmarks, something like

const M_VALUES: [usize; 1] = [1<<7]; 
const N_VALUES: [usize; 1] = [1<<13]; 

will benchmark the cost of doing a two-fold (in the sense of Zip+) linear combination of the rows of a 1<<7 x 1<<13 matrix with:
- Task 1: entries filled with degree 31 binary polynomials (Task 1)
- Task 2: entries filled with 32bit integers (Task 2)
- Task 3: Task 1 but modulo a prime chosen at runtime
- Task 4: Task 2 but modulo a prime chosen at runtime
- Task 1b: Like Task 3 but the coefficients of the polynomials have 2+log(1<<13)=14 bit size (this corresponds to when the coefficient matrix has been encoded and the verifier has to make linear combinations of some of its rows) 