#ifndef SPMV_COMM_HPP
#define SPMV_COMM_HPP

#include "mpi.h"
#include "../par_binary_IO.hpp"
class SpMVComm
{
    public:
        SpMVComm()
        {
            A = NULL;
            sendbuf = NULL;
            recvbuf = NULL;
            spmv_req = NULL;
            n = 0;
            completed_sends = 0;
            completed_recvs = 0;
        }

        ~SpMVComm()
        {
            if (spmv_req)
            {
                delete[] spmv_req;
            }
        }

        void init(ParMat& A_in, double* sbuf, double* rbuf)
        {
            A = &A_in;
            sendbuf = sbuf;
            recvbuf = rbuf;
            n = A->recv_comm.n_msgs + A->send_comm.n_msgs;
            spmv_req = new MPI_Request[n];
        }

        void pack(std::vector<double>& x)
        {
            for (int i = 0; i < A->send_comm.size_msgs; i++)
            {
                sendbuf[i] = x[A->send_comm.idx[i]];
            }
            //put these here because this method is not overridden by children
            completed_sends = 0;
            completed_recvs = 0;
        }

        void start()
        {
            int proc, start, end;
            int tag = 0;

            for (int i = 0; i < A->recv_comm.n_msgs; i++)
            {
                proc  = A->recv_comm.procs[i];
                start = A->recv_comm.ptr[i];
                end   = A->recv_comm.ptr[i + 1];
                MPI_Irecv(&(recvbuf[start]),
                          (int)(end - start),
                          MPI_DOUBLE,
                          proc,
                          tag,
                          MPI_COMM_WORLD,
                          &(spmv_req[i]));
            }

            for (int i = 0; i < A->send_comm.n_msgs; i++)
            {
                proc  = A->send_comm.procs[i];
                start = A->send_comm.ptr[i];
                end   = A->send_comm.ptr[i + 1];
                MPI_Isend(&(sendbuf[start]),
                          (int)(end - start),
                          MPI_DOUBLE,
                          proc,
                          tag,
                          MPI_COMM_WORLD,
                          &(spmv_req[A->recv_comm.n_msgs + i]));
            }

        }

        void wait()
        {
            if (n)
            {
                MPI_Waitall(n, 
                        spmv_req,
                        MPI_STATUSES_IGNORE);
            }
        }

        double* get_recvbuf()
        {
            return recvbuf;
        }

        int test_n_sends(int test_count){
            int offset = A->recv_comm.n_msgs;
            int test = 0;
            int remainder = A->send_comm.n_msgs - completed_sends;
            int use_remainder = test_count>remainder;
            test_count = (!use_remainder)*test_count + use_remainder * remainder;
            MPI_Testall(test_count, &(spmv_req[offset + completed_sends]), &test, MPI_STATUSES_IGNORE);
            completed_sends += test*n;
            return test;
        }
        int test_n_recvs(int test_count){
            int test = 0;
            int remainder = A->recv_comm.n_msgs - completed_recvs;
            int use_remainder = test_count>remainder;
            test_count = (!use_remainder)*test_count + use_remainder * remainder;
            MPI_Testall(test_count, &(spmv_req[completed_recvs]), &test, MPI_STATUSES_IGNORE);
            completed_recvs += test*n;
            return test;
        }
        int get_completed_sends(){return completed_sends;}
        int get_completed_recvs(){return completed_sends;}
        int done(){return (completed_sends+completed_recvs) == n;}

    protected:
        double* sendbuf;
        double* recvbuf;
        int n;
        ParMat* A;
        int completed_sends;
        int completed_recvs;
        
    private:
        MPI_Request* spmv_req;
};

#endif
