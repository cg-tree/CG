#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "spmv.hpp"
#include "mat_ops.hpp"
#include "cg_routines.hpp"
#include <math.h>
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2)
    {
        if (rank == 0) printf("Need a matrix passed in command line\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
        return 1;
    }

    // Init matrix from file
    std::string filename = argv[1];
    ParMat A;
    readParMatrix(filename.c_str(), A);
    form_comm(A);

    // Create linear system
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> r(A.local_rows);  

    //set spmv and inner product
    SPMV_TYPE spmv_type = STANDARD_SPMV;
    auto spmv = get_spmv(spmv_type);
    auto standard_inner_product = get_inner_product(STANDARD_INNER_PRODUCT);
    
    // Set x to random values, b = A*x 
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), [&]() { return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b);
    MPI_Request request;
    double tmp, bb_inner, rr_inner;
    standard_inner_product(b, b, &tmp,&bb_inner,&request);
    double norm_b = sqrt(bb_inner);
    double norm_r;

    // Declare book keeping variables
    int n_iter, n_tests;
    double t0,t1,time;
    int max_iters = 500;
    double tol = 1e-6;

    /*********************************************************
     * Test Standard CG 
     ********************************************************/

    if (rank == 0) 
        printf("Testing Standard CG:\n");
    auto standard_cg = get_cg(STANDARD_CG);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = standard_cg(A, x, b,max_iters,tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);
    
    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test Preconditioned CG
     ********************************************************/
    if (rank == 0) 
        printf("Testing Preconditioned CG:\n");

    auto preconditioned_cg = get_cg(PRECONDITIONED_CG,STANDARD_SPMV,STANDARD_INNER_PRODUCT);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = preconditioned_cg(A, x, b,max_iters,tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e, raw res is %e, norm_b is %e\n", n_iter, norm_r / norm_b,norm_r,norm_b);


    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test Pipe CG
     ********************************************************/
    if (rank == 0) 
        printf("Testing Pipe CG:\n");

    auto pipe_cg = get_cg(PIPE_CG,STANDARD_SPMV, STANDARD_INNER_PRODUCT);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = pipe_cg(A, x, b,max_iters, tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    tmp =0;
    rr_inner=0;
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e, raw res is %e\n", n_iter, norm_r / norm_b,norm_r);
        //printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);

    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test Pipe CG with Non-Blocking InnerProduct
     ********************************************************/
    if (rank == 0)
        printf("Testing Pipe CG with Non-Blocking InnerProduct:\n");

    pipe_cg = get_cg(PIPE_CG,STANDARD_SPMV,NON_BLOCKING_INNER_PRODUCT);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = pipe_cg(A, x, b,max_iters,tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e, raw res is %e\n", n_iter, norm_r / norm_b,norm_r);
        //printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test Pipe CG with PROGRESS_SPMV and standard inner product
     ********************************************************/
    if (rank == 0)
        printf("Testing Pipe CG with PROGRESS_SPMV:\n");

    pipe_cg = get_cg(PIPE_CG,PROGRESS_SPMV,STANDARD_INNER_PRODUCT);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = pipe_cg(A, x, b,max_iters,tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e, raw res is %e\n", n_iter, norm_r / norm_b,norm_r);
        //printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test Pipe CG with PROGRESS_SPMV and NON_BLOCKING inner product
     ********************************************************/
    if (rank == 0)
        printf("Testing Pipe CG with PROGRESS_SPMV and NON_BLOCKING_INNER_PRODUCT:\n");

    pipe_cg = get_cg(PIPE_CG,PROGRESS_SPMV,NON_BLOCKING_INNER_PRODUCT);
    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    t0 = MPI_Wtime();
    n_iter = pipe_cg(A, x, b,max_iters,tol);
    t1 = MPI_Wtime();
    t1 -= t0;
    r = b;
    spmv(1.0, A, x, -1.0, r);
    standard_inner_product(r, r,&tmp,&rr_inner,&request);
    norm_r = sqrt(rr_inner);
    if (rank == 0) 
        printf("After %d iterations, final res is %e, raw res is %e\n", n_iter, norm_r / norm_b,norm_r);
        //printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    MPI_Allreduce(&t1,&time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    MPI_Finalize();

    return 0;
}
