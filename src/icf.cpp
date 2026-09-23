#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "sparse_mat.hpp"
#include "mat_ops.hpp"
#include "icf.hpp"


int set_A_equals_LLT(Mat& A, Mat& L){
    //set A to be LL^T because A.on_proc may be singular 
    for(int i = 0; i< A.n_rows;++i){
        int start,end;
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for(int j=start; j< end;++j){
            A.data[j] = sparse_row_partial_inner_product(L,i,A.col_idx[j],L.n_cols);
        }
    }

}
int test_A_equals_LLT(Mat& A, Mat& L, double tol){
    int errorcount = 0;
    double errorsum = 0;
    for(int i = 0; i< A.n_rows;++i){
        int start,end;
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for(int j=start; j< end;++j){
            double aij_approx = sparse_row_partial_inner_product(L,i,A.col_idx[j],L.n_cols);
            
            if( fabs(aij_approx - A.data[j]) > tol ){
                ++errorcount;
                errorsum += fabs(aij_approx - A.data[j]);
            }
        }
    }
    printf("total absolute error %f\n", errorsum);
    return errorcount;
}

int test_Ax_equals_b(Mat& A, std::vector<double> x, std::vector<double>b, double tol){
    
    int errorcount = 0;
    double errorsum = 0;
    double max_error = 0;
    for(int i = 0; i< A.n_rows;++i){
        double bi = sparse_row_vector_inner_product(A,i,x);
        double error = fabs(bi - b[i]);
        if( error > tol ){
            ++errorcount;
            errorsum += error;
        }
        if(error>max_error){max_error = error;}
    }
    printf("total absolute error %f, max error %f\n", errorsum, max_error);
    return errorcount;
}
int test_LTx_equals_y(Mat& A, std::vector<double> x, std::vector<double> y, double tol){
    auto tmp = std::vector<double>(y.size(),0);
    int errorcount = 0;
    double errorsum = 0;
    double max_error = 0;
    for(int i = 0; i< A.n_rows;++i){
        int col = i;//bc transpose
        double start,end;
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for(int j = start; j<end; ++j){
            int row = A.col_idx[j];//bc of transpose
            tmp[row] += x[col]* A.data[j];
        }
    }

    for(int i = 0; i< A.n_rows;++i){
        double error = fabs(y[i] - tmp[i]);
        if( error > tol ){
            ++errorcount;
            errorsum += error;
        }
        if(error>max_error){max_error = error;}
    }
    printf("total absolute error %f, max error %f\n", errorsum, max_error);
    return errorcount;

}

int test_mat_equals_mat(Mat& A, Mat& L){

    int nnz = A.nnz == L.nnz;
    int ncol = A.n_cols == L.n_cols;
    int nrow = A.n_rows == L.n_rows;
    int rowptr = A.rowptr.data() != L.rowptr.data();
    int rowptrsize = A.rowptr.size() == L.rowptr.size();
    int colidx = A.col_idx.data() != L.col_idx.data();
    int colidxsize = A.col_idx.size() == L.col_idx.size();
    int data = A.data.data() != L.data.data();
    int datasize = A.data.size() == L.data.size();
    
    int pass = nnz && ncol && nrow && rowptr && colidx && data && rowptrsize && colidxsize && datasize;
    return pass;
}
/*
 * sqrt( A_{i,i} - |L_{i,:i-1}|^2 )
 * */
double get_lii(Mat& A, Mat& L, int i)
{
    int start,end;
    start = A.rowptr[i];
    end = A.rowptr[i+1];
    int col_idx_i = start;
    double aii;
    for(int j = start; (A.col_idx[j] <= i)&&(j < end); ++j)
    {
        col_idx_i = j;
    }
    aii = A.data[col_idx_i];
    double ldotl = sparse_row_partial_inner_product(L,i,i,i);
    double lii = sqrt(aii - ldotl);
    L.data[col_idx_i] = lii;
    return lii;
}

inline void get_lji(Mat& A, Mat& L, int i, int j, double lii)
{
    double aji;
    int indexji =0;
    for(int k = A.rowptr[j];
        (k<A.rowptr[j+1]) && (A.col_idx[k]<=i);
        ++k)
    {
        indexji = k;
    }
    if(A.col_idx[indexji] != i){
        return;
    }
    aji = A.data[indexji];
    if(aji == 0){
        L.data[indexji] = 0;
        return;
    }
    double lidotlj = sparse_row_partial_inner_product(L,i,j,i);
    double lji = (aji - lidotlj) / lii;
    L.data[indexji] = lji;
    return;
}

inline void for_lji(Mat& A, Mat& L, int i, double lii){

    for(int j = i+1; j < A.n_rows; ++j)
    {
        get_lji(A,L,i,j,lii);
    }
}


void setup_L(Mat& A, Mat& L){
    L.rowptr = std::vector<int>(A.rowptr.begin(),A.rowptr.end());
    L.col_idx = std::vector<int>(A.col_idx.begin(),A.col_idx.end());
    L.data = std::vector<double>(A.data.size(),0);
    L.n_rows = A.n_rows;
    L.n_cols= A.n_cols;
    L.nnz = A.nnz;
}

void incomplete_cholesky_factor(Mat& A, Mat& L)
{
    double lii_time = 0;
    double lji_time = 0;
    
    for(int i = 0; i < L.n_rows; ++i)
    {
        double t0 = MPI_Wtime();
        double lii = get_lii(A,L,i);
        double t1 = MPI_Wtime();
        lii_time += t1 - t0;
        for_lji(A,L,i,lii);
        double t2 = MPI_Wtime();
        lji_time+= t2 - t1;
    }
    printf("lii time %fs lji time %fs\n",lii_time, lji_time);

}

void incomplete_cholesky(Mat& A, Mat& L){
    setup_L(A,L);
    incomplete_cholesky_factor(A,L);
}
void incomplete_cholesky(ParMat& A, Mat& L){ incomplete_cholesky(A.on_proc, L); }

void incomplete_cholesky_solve(Mat& L, std::vector<double>& x, std::vector<double>& b){
    auto y = std::vector<double>(x.size(),0);
    forward_solve(L,y,b);
    backward_solve(L,x,y);
}

void test_incomplete_cholesky(Mat& A, std::vector<double>& x, std::vector<double>& b)
{
    double t0 = MPI_Wtime();
    Mat L;
    setup_L(A,L);
    double t1 = MPI_Wtime();

    incomplete_cholesky_factor(A,L);//A might might not be factorizable
    double t2 = MPI_Wtime();
    set_A_equals_LLT(A,L);//set A to be factorizable
    double t3 = MPI_Wtime();
    //incomplete_cholesky_factor(A,L);//compute factorization that should succeed
    double t4 = MPI_Wtime();
    //printf("L alloc time %fs\n", t1 - t0);
    //printf("L factor time %fs\n", t2 - t1);
    //printf("set A=LL^T time %fs\n", t3 - t2);

    printf("cholesky had %d values that differed by more than tolerance\n",test_A_equals_LLT(A,L,1e-12));//test if factorization succeeds
    
    //solve
    incomplete_cholesky_solve(L,x,b);

    auto y = std::vector<double>(x.size(),0);
    forward_solve(L,y,b);
    printf("forward solve had %d values that differed by more than tolerance\n",
    test_Ax_equals_b(L,y,b, 1e-12));
    backward_solve(L,x,y);
    printf("backward solve had %d values that differed by more than tolerance\n",
    test_LTx_equals_y(L,x,y, 1e-12));
    //test that the solve actually works here vv
    printf("cholesky solve had %d values that differed by more than tolerance\n",
    test_Ax_equals_b(A,x,b, 1e-9));
}
void test_incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    test_incomplete_cholesky(A.on_proc, x, b);
}

