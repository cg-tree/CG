#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include <math.h>
#include "comm/comm_routines.hpp"
#include "spmv.hpp"
#include "mat_ops.hpp"
#include "icf.hpp"
#include "cg_routines.hpp"

int preconditioned_cg(SPMV_TYPE spmv_type, INNER_PRODUCT_TYPE inner_product_type, ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b, 
        int max_iters = 500,
        double tol = 1e-06){

    auto spmv = get_spmv(spmv_type);
    auto inner_product = get_inner_product(inner_product_type);
    auto inner_product_wait = get_inner_product_wait(inner_product_type);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    std::vector<double> res;
    double time_start = MPI_Wtime();
    
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
    double norm_r;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);

    Mat L;
    incomplete_cholesky(A,L);
    incomplete_cholesky_solve(L,z,r);

    // p0 = z0
    p = z;

    // Find initial (r, r) and residual
    double sum_local;
    MPI_Request request;
    inner_product(r, z,&sum_local,&rz_inner,&request);
    inner_product_wait(&request, MPI_STATUS_IGNORE);
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
    while (norm_r > tol && iter < max_iters)
    {
        double t2 = MPI_Wtime();
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap);
        //App_inner = inner_product(Ap, p);
        inner_product(Ap, p, &sum_local, &App_inner, &request);
        inner_product_wait(&request, MPI_STATUS_IGNORE);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rz_inner / App_inner;
        double t3 = MPI_Wtime();
        //printf("update alpha time %fs\n",t3 - t2);

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

        //next_inner = inner_product(r, z);
        inner_product(r, z,&sum_local,&next_inner,&request);
        inner_product_wait(&request, MPI_STATUS_IGNORE);

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
    /*
    if (rank == 0) 
    {
        double time_end = MPI_Wtime();
        printf("time elapsed:%f seconds\n",time_end - time_start);
        if (iter == max_iters)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }*/
    return iter;
}

int standard_cg(SPMV_TYPE spmv_type, INNER_PRODUCT_TYPE inner_product_type, ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b, 
        int max_iters = 500,
        double tol = 1e-06)
{
    auto spmv = get_spmv_advance(spmv_type);
    auto inner_product = get_inner_product_advance();

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    // Setup persistent allreduces
    double local_sum, global_sum;
    SpMVComm* spmv_comm = new SpMVComm();
    AllreduceComm* allreduce_comm = new AllreduceComm();

    spmv_comm->init(A, sendbuf.data(), recvbuf.data());
    allreduce_comm->init(&local_sum, &global_sum);

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r;
    // int max_iter = ((int)(1.3*b.size())) + 2;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r, spmv_comm);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r, allreduce_comm);

    norm_r = sqrt(rr_inner);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter        = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iters)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap, spmv_comm);
        App_inner = inner_product(Ap, p, allreduce_comm);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0 * alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r, spmv_comm);
        }

        next_inner = inner_product(r, r, allreduce_comm);
        beta       = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r   = sqrt(rr_inner);

        iter++;
    }

    delete spmv_comm;
    delete allreduce_comm;

    return iter;
}


int pipe_cg(SPMV_TYPE spmv_type, INNER_PRODUCT_TYPE inner_product_type, ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b, 
        int max_iters = 500,
        double tol = 1e-06)
{
    
    auto spmv = get_spmv(spmv_type);
    auto inner_product = get_inner_product(inner_product_type);
    auto combined_inner_product = get_combined_inner_product(inner_product_type);
    auto inner_product_wait = get_inner_product_wait(inner_product_type); 
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double time_start = MPI_Wtime();
    std::vector<double> res;
    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> w(A.local_rows);
    std::vector<double> s(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> z(A.local_rows);
    std::vector<double> q(A.local_rows);
    std::vector<double> u(A.local_rows);
    std::vector<double> m(A.local_rows);
    std::vector<double> n(A.local_rows);



    int iter, recompute_r;
    double alpha, alpha_last;
    double beta, delta;
    double gamma, gamma_last;
    double gamma_delta[2];
    double sum_local[2];
    double rr_inner;
    double norm_r;

    Mat L;
    incomplete_cholesky(A,L);
    //incomplete_cholesky_solve(L,x,b);

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);
    MPI_Request rr_inner_request;
    //Iinner_product(r,r,sum_local,&rr_inner,&rrinner_request);
    inner_product(r,r,sum_local,&rr_inner,&rr_inner_request);
    incomplete_cholesky_solve(L,u,r);

    //w_0 = Au_0
    spmv(1.0, A, u, 0.0, w);

    //MPI_Wait(&rrinner_request, MPI_STATUS_IGNORE);
    inner_product_wait(&rr_inner_request, MPI_STATUS_IGNORE);
    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);
    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iters)
    {
        //non-blocking dot products
        //gamma_i = (r_i,u_i)
        //delta = (w_i, u_i)
        MPI_Request gamma_delta_request;
        //combined_Iinner_product(r,w,u, sum_local, gamma_delta,&gamma_delta_request);
        combined_inner_product(r,w,u, sum_local, gamma_delta,&gamma_delta_request);

        //main computational load
        incomplete_cholesky_solve(L,m,w);
        //n_i = Am_i
        spmv(1.0, A, m, 0.0, n);

        //wait for dot product results
        //MPI_Wait( &gamma_delta_request, MPI_STATUS_IGNORE );
        inner_product_wait( &gamma_delta_request, MPI_STATUS_IGNORE );
        
        gamma = gamma_delta[0];
        delta = gamma_delta[1];
     
        //scalar updates
        if(iter > 0)
        {
            beta = gamma / gamma_last;
            alpha = gamma / ( delta - (beta * gamma / alpha_last) );
        }
        else
        {
            beta = 0;
            alpha = gamma/delta;
        }
        //vector updates
        
        //z_i = n_i + beta_i z_{i-1}
        scale(beta,z);
        axpy(1.0,z,n);
        //q_i = m_i + beta_i q_{i-1}
        scale(beta,q);
        axpy(1.0,q,m);
        //s_i = w_i + beta_i s_{i-1}
        scale(beta,s);
        axpy(1.0,s,w);
        //p_i = u_i + beta_i p_{i-1}
        scale(beta,p);
        axpy(1.0,p,u);

        //x_{i+1} = x_i + alpha_i p_i
        axpy(alpha, x, p);
        //u_{i+1} = u_i - alpha_i q_i
        axpy(-1.0*alpha, u, q);
        //w_{i+1} = w_i - alpha_i z_i
        axpy(-1.0*alpha, w, z);
        //r_{i+1} = r_i - alpha_i s_i
        axpy(-1.0*alpha, r, s);
        
        //progress tracking
        norm_r = sqrt(gamma);
        res.push_back(norm_r);

        //set scalar last values
        gamma_last = gamma;
        alpha_last = alpha;

        iter++;
        //end
    }
/*   
    if (rank == 0) 
    {
        double time_end = MPI_Wtime();
        printf("time elapsed:%f seconds\n",time_end - time_start);
        if (iter == max_iters)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
     double total_test_time = get_total_test_time();
        double total_naive_test_time = get_naive_total_test_time();
        int total_test_count = get_total_test_count();
        printf("rank0 total_test_time:%f rank0 avg_test_time:%f\n",
            total_test_time,total_test_time / (double) total_test_count);
        printf("rank0 total_test_time:%f total_naive_test_time:%f %fs not counted due to catastrophic loss of precision\n",
            total_test_time, total_naive_test_time, total_test_time - total_naive_test_time);
    }
    */
    return iter;
}




std::function<int(ParMat&, 
        std::vector<double>&, 
        std::vector<double>&, 
        int,
        double)> get_cg( CG_TYPE cg_type, SPMV_TYPE spmv_type, INNER_PRODUCT_TYPE inner_product_type){
    switch(cg_type){
        
      case STANDARD_CG:
            return std::bind(standard_cg, spmv_type, inner_product_type, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5);
        
        case PRECONDITIONED_CG:
            return std::bind(preconditioned_cg, spmv_type, inner_product_type, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5);
        
        case PIPE_CG:
            return std::bind(pipe_cg, spmv_type, inner_product_type, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5);
    };
}
