#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include <math.h>
#include "spmv.hpp"
#include "mat_ops.hpp"
#include "icf.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    double time_start = MPI_Wtime();

    const char* filename = "Dubcova2.pm";
    if (argc > 1)
    {
        filename = argv[1];
    }

    ParMat A;
    readParMatrix(filename, A);
    form_comm(A);
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> res;

    // Set b to random values, x to 0
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), 
            [&](){ return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b);
    std::fill(x.begin(), x.end(), 0);

    
#ifdef TEST
    test_incomplete_cholesky(A,x,b);
#endif// TEST
    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> z(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);

    int iter, recompute_r;
    double alpha, beta;
    double rz_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    int max_iter = ((int)(1.3*b.size())) + 2;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);

    Mat L;
    incomplete_cholesky(A,L);
    incomplete_cholesky_solve(L,z,r);

    // p0 = z0
    p = z;

    // Find initial (r, r) and residual
    rz_inner = inner_product(r, z);
    norm_r = sqrt(rz_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        double t2 = MPI_Wtime();
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap);
        App_inner = inner_product(Ap, p);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rz_inner / App_inner;
        double t3 = MPI_Wtime();
        printf("update alpha time %fs\n",t3 - t2);

        // x_{i+1} = x_i + alpha_i * p_i
        axpy(alpha, x, p);

        
        if ((iter % recompute_r) && iter > 0) // don't recompute r
        {
            axpy(-1.0*alpha, r, Ap);
        }
        else //recompute r
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r);
        }

        incomplete_cholesky_solve(L,z,r);

        next_inner = inner_product(r, z);
        beta = next_inner / rz_inner;

        scale(beta, p);
        //p_{i+1} = p_{i} + z_{i+1}
        axpy(1.0, p, z);

        // Update next inner product
        rz_inner = next_inner;
        norm_r = sqrt(rz_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0) 
    {
        double time_end = MPI_Wtime();
        printf("time elapsed:%f seconds\n",time_end - time_start);
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }

    MPI_Finalize();
}
