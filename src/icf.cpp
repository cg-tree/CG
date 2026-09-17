#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "sparse_mat.hpp"
#include "mat_ops.hpp"
#include "icf.hpp"


int test_cholesky(Mat& A, Mat& L, double tol){

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
    aji = A.data[indexji];
    double lji = (aji - lidotlj) / lii;
    L.data[lji] = lji;
    return lji;
}

void incomplete_cholesky(Mat& A, std::vector<double>& x, std::vector<double>& b)
{
    Mat l;
    l.rowptr = std::vector<int>(A.rowptr.begin(),A.rowptr.end());
    l.col_idx = std::vector<int>(A.col_idx.begin(),A.col_idx.end());
    l.data = std::vector<double>(A.data.size(),0);
    l.n_rows = A.n_rows;
    l.n_cols= A.n_cols;
    l.nnz = A.nnz;

    if(!test_mat_equals_mat(A,l)){
        printf("mat not equals mat\n");return;}
    for(int i = 0; i < l.n_rows; ++i)
    {
        double lii = get_lii(A,l,i);
        for(int j = i+1; j < l.n_rows; ++j)
        {
            get_lji(A,l,i,j,lii);
        }
    }

    printf("cholesky had %d values that differed by more than tolerance\n",test_cholesky(A,l,1e-5));
}

void incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    incomplete_cholesky(A.on_proc, x, b);
}

