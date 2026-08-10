#include "sparse_mat.hpp"
#include "spmv.hpp"
#include <math.h>
static double total_test_time = 0;
static double sum_test_end_time = 0;
static double sum_test_start_time = 0;
static int total_test_count = 0;
double get_total_test_time(){ return sum_test_end_time - sum_test_start_time;}
double get_naive_total_test_time(){ return total_test_time;}
int get_total_test_count(){ return total_test_count;}
void spmv_row_partial(int row,int colstart,int colend, double alpha,
        Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b)
{
    double sum;
    int start, end;
    start = A.rowptr[row];
    end = A.rowptr[row+1];
    sum = 0;
    for (int j = start; j < end; j++)
    {
        if((colstart <= A.col_idx[j]) && (colend > A.col_idx[j]) )
            sum += A.data[j] * x[A.col_idx[j]];
    }
    b[row] = alpha * sum + beta * b[row];
}

//1 row of spmv
void spmv_row(int row, double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b)
{
    double sum;
    int start, end;
    start = A.rowptr[row];
    end = A.rowptr[row+1];
    sum = 0;
    for (int j = start; j < end; j++)
    {
        sum += A.data[j] * x[A.col_idx[j]];
    }
    b[row] = alpha * sum + beta * b[row];
}

//not particularly useful
int spmv_testall(int nmsgs, MPI_Request* requests, double alpha, Mat& A,
        std::vector<double>& x, double beta, std::vector<double>& b)
{
    int test;
    for (int i = 0; i < A.n_rows; i++){
        spmv_row(i, alpha, A, x, beta, b);
        MPI_Testall(nmsgs, requests, &test, MPI_STATUSES_IGNORE);
    }
    return test;
}

/* call after initiating isends and irecvs
 * tracks time spent in test calls
 * */
void spmv_test(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, std::vector<double>& recvbuf)
{
    double time_start, time_end;
    int next_request=0;
    int nmsgs = A.recv_comm.n_msgs;

    MPI_Request* requests = A.recv_comm.req.data();
    for (int i = 0; i < A.on_proc.n_rows; i++){
        int test = 0;
        spmv_row(i, alpha, A.on_proc, x, beta, b);
        if(next_request < nmsgs)
        {
            time_start = MPI_Wtime();
            MPI_Test(&(requests[next_request]), &test, MPI_STATUS_IGNORE);
            time_end = MPI_Wtime();
            
            sum_test_start_time += time_start;
            sum_test_end_time += time_end;
            
            total_test_time += time_end - time_start;
            total_test_count++;
        }
        if(test)
        {
            int start,end; 
            start = A.recv_comm.ptr[next_request];
            end   = A.recv_comm.ptr[next_request + 1];
            spmv_row_partial(next_request, start, end, alpha, A.off_proc, recvbuf, 1.0, b);
            next_request++;
        }
    }
    while(next_request < nmsgs)
    {
        int test = 0;
        if(next_request < nmsgs)
        {
            time_start = MPI_Wtime();
            MPI_Test(&(requests[next_request]), &test, MPI_STATUS_IGNORE);
            time_end = MPI_Wtime();
            
            sum_test_start_time += time_start;
            sum_test_end_time += time_end;
            
            total_test_time += time_end - time_start;
            total_test_count++;
        }
        if(test)
        {
            int start,end; 
            start = A.recv_comm.ptr[next_request];
            end   = A.recv_comm.ptr[next_request + 1];
            spmv_row_partial(next_request, start, end, alpha, A.off_proc, recvbuf, 1.0, b);
            next_request++;
        }
    }
}
/* call after initiating isends and irecvs
 * tracks time spent in test calls
 * checks for messages received in reverse order because we
 * want to know to what degree the perfomance of spmv_test is
 * due order of memory accesses
 * */
void spmv_test_reversed(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, std::vector<double>& recvbuf)
{
    double time_start, time_end;
    int nmsgs = A.recv_comm.n_msgs;
    int next_request = nmsgs - 1;

    MPI_Request* requests = A.recv_comm.req.data();
    for (int i = 0; i < A.on_proc.n_rows; i++){
        int test = 0;
        spmv_row(i, alpha, A.on_proc, x, beta, b);
        if(next_request >= 0)
        {
            time_start = MPI_Wtime();
            MPI_Test(&(requests[next_request]), &test, MPI_STATUS_IGNORE);
            time_end = MPI_Wtime();
            
            sum_test_start_time += time_start;
            sum_test_end_time += time_end;
            
            total_test_time += time_end - time_start;
            total_test_count++;
        }
        if(test)
        {
            int start,end; 
            start = A.recv_comm.ptr[next_request];
            end   = A.recv_comm.ptr[next_request + 1];
            spmv_row_partial(next_request, start, end, alpha, A.off_proc, recvbuf, 1.0, b);
            next_request--;
        }
    }
    while(next_request >= 0)
    {
        int test = 0;
        if(next_request >= 0)
        {
            time_start = MPI_Wtime();
            MPI_Test(&(requests[next_request]), &test, MPI_STATUS_IGNORE);
            time_end = MPI_Wtime();
            
            sum_test_start_time += time_start;
            sum_test_end_time += time_end;
            
            total_test_time += time_end - time_start;
            total_test_count++;
        }
        if(test)
        {
            int start,end; 
            start = A.recv_comm.ptr[next_request];
            end   = A.recv_comm.ptr[next_request + 1];
            spmv_row_partial(next_request, start, end, alpha, A.off_proc, recvbuf, 1.0, b);
            next_request--;
        }
    }
}


// Serial SpMV b = alpha*A*x + eta*b
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
// optimized spmv1
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

    //spmv(alpha, A.on_proc, x, beta, b);
    //spmv_test(alpha, A, x, beta, b, recvbuf);
    spmv_test_reversed(alpha, A, x, beta, b, recvbuf);

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }
    
}


