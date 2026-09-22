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
        aii = A.data[j];
        col_idx_i = j;
    }
    double ldotl = sparse_row_partial_inner_product(L,i,i,i);
    double lii = sqrt(aii - ldotl);
    L.data[col_idx_i] = lii;
    return lii;
}

double get_lji(Mat& A, Mat& L, int i, int j, double lii)
{
    double lidotlj = sparse_row_partial_inner_product(L,i,j,i);
    double aji;
    int indexji =0;
    for(int k = A.rowptr[j];
        (k<A.rowptr[j+1]) && (A.col_idx[k]<=i);
        ++k)
    {
        indexji = k;
    }
    if(A.col_idx[indexji] != i){
        return 0;
    }
    aji = A.data[indexji];
    double lji = (aji - lidotlj) / lii;
    L.data[indexji] = lji;
    return lji;
}

void test_incomplete_cholesky(Mat& A, std::vector<double>& x, std::vector<double>& b)
{
    double t0 = MPI_Wtime();
    Mat l;
    l.rowptr = std::vector<int>(A.rowptr.begin(),A.rowptr.end());
    l.col_idx = std::vector<int>(A.col_idx.begin(),A.col_idx.end());
    l.data = std::vector<double>(A.data.size(),0);
    l.n_rows = A.n_rows;
    l.n_cols= A.n_cols;
    l.nnz = A.nnz;
    double t1 = MPI_Wtime();
    //if(!test_mat_equals_mat(A,l)){
    //    printf("mat not equals mat\n");return;
    //}

    incomplete_cholesky(A,l,x,b);//A might might not be factorizable
    double t2 = MPI_Wtime();
    set_A_equals_LLT(A,l);//set A to be factorizable
    double t3 = MPI_Wtime();
    incomplete_cholesky(A,l,x,b);//compute factorization that should succeed
    double t4 = MPI_Wtime();
    printf("L alloc time %fs\n", t1 - t0);
    printf("L factor time %fs\n", t2 - t1);
    printf("set A=LL^T time %fs\n", t3 - t2);

    printf("cholesky had %d values that differed by more than tolerance\n",test_A_equals_LLT(A,l,1e-9));//test if factorization succeeds

}
void incomplete_cholesky(Mat& A, Mat& L, std::vector<double>& x, std::vector<double>& b)
{
    double lii_time = 0;
    double lji_time = 0;
    
    for(int i = 0; i < L.n_rows; ++i)
    {
        double t0 = MPI_Wtime();
        double lii = get_lii(A,L,i);
        double t1 = MPI_Wtime();
        lii_time += t1 - t0;
        for(int j = i+1; j < L.n_rows; ++j)
        {
            get_lji(A,L,i,j,lii);
        }
        double t2 = MPI_Wtime();
        lji_time+= t2 - t1;
    }
    printf("lii time %fs lji time %fs\n",lii_time, lji_time);

}

void incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    test_incomplete_cholesky(A.on_proc, x, b);
}

