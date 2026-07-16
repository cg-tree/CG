#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"

// Serial SpMV b = alpha*A*x + beta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b)
{
    double sum;
    int start, end;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        sum = 0;
        for (int j = start; j < end; j++)
        {
            sum += A.data[j] * x[A.col_idx[j]];
        }
        b[i] = alpha * sum + beta * b[i];
    }
}

// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(&(recvbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        for (int j = start; j < end; j++)
            sendbuf[j] = x[A.send_comm.idx[j]];
        MPI_Isend(&(sendbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.send_comm.req[i]));
    }

    spmv(alpha, A.on_proc, x, beta, b);

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    spmv(alpha, A.off_proc, recvbuf, 1.0, b);

}

double inner_product(std::vector<double> a, std::vector<double> b)
{
    double sum, sum_local;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
        sum_local += a[i] * b[i];

    MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}

void Iinner_product(std::vector<double> a, std::vector<double>b, double* sum, MPI_Request* request)
{
    double sum_local;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
        sum_local += a[i] * b[i];

    MPI_Iallreduce(&sum_local, sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, request);
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> w(A.local_rows);
    std::vector<double> s(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> z(A.local_rows);
    std::vector<double> q(A.local_rows);



    int iter;
    double alpha, alpha_last;
    double beta, delta;
    double gamma, gamma_last;
    double rr_inner;
    double norm_r, tol = 1e-6;
    int max_iter = ((int)(1.3*b.size())) + 2;
    max_iter = 200;    

    // r0 = b - A * x0
    //x0 = 0 so r0 = b
    scale(0.0,r);
    axpy(1.0,r,b);

    
    //w_0 = Ar_0
    spmv(1.0, A, r, 0.0, w);

    rr_inner = inner_product(r,r);
    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);
    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        //non-blocking dot products
        //gamma_i = (r_i,u_i)
        MPI_Request gamma_request;
        //Iinner_product( r, u, &gamma, &gamma_request );
        Iinner_product( r, r, &gamma, &gamma_request );
        //delta = (w_i, u_i)
        MPI_Request delta_request;
        //Iinner_product( w, u, &delta, &delta_request );
        Iinner_product( w, r, &delta, &delta_request );

        //main computational load
        //q_i = Aw_i
        spmv(1.0, A, w, 0.0, q);

        //wait for dot product results
        MPI_Wait( &gamma_request, MPI_STATUS_IGNORE );
        MPI_Wait( &delta_request, MPI_STATUS_IGNORE );

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
        
        //z_i = q_i + beta_i z_{i-1}
        scale(beta,z);
        axpy(1.0,z,q);
        //s_i = w_i + beta_i s_{i-1}
        scale(beta,s);
        axpy(1.0,s,w);
        //p_i = r_i + beta_i p_{i-1}
        scale(beta,p);
        axpy(1.0,p,r);

        //x_{i+1} = x_i + alpha_i p_i
        axpy(alpha, x, p);
        //r_{i+1} = r_i - alpha_i s_i
        axpy(-1.0*alpha, r, s);
        //w_{i+1} = w_i - alpha_i z_i
        axpy(-1.0*alpha, w, z);

        //progress tracking
        rr_inner = inner_product(r,r);
        norm_r = sqrt(rr_inner);
        res.push_back(norm_r);

        //set scalar last values
        gamma_last = gamma;
        alpha_last = alpha;

        iter++;
        //end
    }

    if (rank == 0) 
    {
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }

    MPI_Finalize();
}
