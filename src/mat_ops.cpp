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
    
    for( ;(A.col_idx[ia] < colend) && (A.col_idx[ib] < colend) &&
        (ia < enda)&&(ib < endb); )
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

double sparse_row_vector_inner_product(Mat& A, int rowa, std::vector<double> x){
    double sum = 0;
    int start, end;
    start = A.rowptr[rowa];
    end = A.rowptr[rowa+1];
    for(int i = start; i< end; ++i){
        sum += A.data[i] * x[A.col_idx[i]];
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

/*
 * Solves Ly = b
 * reference: https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html
 * */
void forward_solve(Mat& L, std::vector<double>& y, std::vector<double>& b){
    int start,end;
    for(int i = 0; i<L.n_rows; ++i){
        start = L.rowptr[i];
        end = L.rowptr[i+1];
        double tmp = b[i];
        double lii = 1;
        for(int j = start; j< end; ++j){
            int col = L.col_idx[j];
            tmp -= L.data[j] * y[col];
            if(i == col){
              lii = L.data[j];
              j = end;//exit loop
            }
        }
        y[i] = tmp / lii;
    }
}

/*
 * Solves L^Tx = b
 * expects: L to be lower triangular
 * 
 * reference: https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-9-linsys.html
 *
 * */
void backward_solve(Mat& L, std::vector<double>& x, std::vector<double>& b){
    int start,end;
    std::vector<double> tmp = b;
    double ljj;
    int i;
    //auto sums = std::vector<double>(b.size(),0);
    for(int j = L.n_rows -1; j>=0; --j){
        start = L.rowptr[j];
        end = L.rowptr[j+1];
        for(i = end -1; (L.col_idx[i]>j) && (i >= start); --i){}
        ljj = L.data[i];
        //double tmpj = b[j] - sums[j];
        double xj = tmp[j] / ljj;
        //double xj = tmpj / ljj;
        x[j] = xj;
        for(--i; i >= start; --i){
            int col = L.col_idx[i];
            tmp[col] -= L.data[i] * xj;
            //sums[col] += L.data[i] * xj;
        }
    }
}

