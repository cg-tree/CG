#include <vector>
#include "sparse_mat.hpp"
#include "mat_ops.hpp"
double inner_product(std::vector<double> a, std::vector<double> b)
{
    double sum, sum_local;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
        sum_local += a[i] * b[i];

    MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}

double inner_product(std::vector<double> a, std::vector<double>b, int is, int ie)
{
    
    double sum_local = 0;
    for (int i = is; i < ie; i++)
        sum_local += a[i] * b[i];
    return sum_local;
}

double sparse_row_partial_inner_product(Mat& A, int rowa,int rowb,int colend )
{
    double sum = 0;
    int starta, startb, enda, endb;
    starta = A.rowptr[rowa];
    enda = A.rowptr[rowa+1];
    startb = A.rowptr[rowb];
    endb = A.rowptr[rowb+1];
    int ia = starta;
    int ib = startb;
    if(A.col_idx[enda-1] < colend){ colend = A.col_idx[enda-1]+1; }
    if(A.col_idx[endb-1] < colend){ colend = A.col_idx[endb-1]+1; }
    for(;(A.col_idx[ia] < colend) && (A.col_idx[ib] < colend);)
    {
        if(A.col_idx[ia] == A.col_idx[ib])
        {
            sum += A.data[ia] * A.data[ib];
            ++ia;
            ++ib;
        }
        else if(A.col_idx[ia] < A.col_idx[ib])
        {
            ++ia;
        }
        else
        {
            ++ib;
        }
        
    }
    return sum;
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y, int is, int ie)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}

void scale(double alpha, std::vector<double>& x, int is, int ie)
{
    for (int i = is; i < ie; i++)
        x[i] = alpha*x[i];
}

