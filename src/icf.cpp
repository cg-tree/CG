#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "sparse_mat.hpp"
#include "mat_ops.hpp"
#include "icf.hpp"


int test_cholesky(Mat& A, Mat& L, double tol){

    int errorcount = 0;
    for(int i = 0; i< A.n_rows;++i){
        int start,end;
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        for(int j=start; j< end;++j){
            double aij_approx = sparse_row_partial_inner_product(L,i,A.col_idx[j],L.n_cols);
            
            if( fabs(aij_approx - A.data[j]) > tol ){
                ++errorcount;
            }
        }
    }
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
    //printf("lii %d\n", i);
    int start,end;
    start = A.rowptr[i];
    end = A.rowptr[i+1];
    int col_idx_i = start;
    double aii;
    //printf("start%d end%d \n",start,end);
    for(int j = start; (A.col_idx[j] <= i)&&(j < end); ++j)
    {
        aii = A.data[j];
        col_idx_i = j;
    }
    //    printf("col_idx_i = %d\n",col_idx_i);
    double ldotl = 0;
    for(int k = start; k < col_idx_i; ++k){
        ldotl += L.data[k] * L.data[k];
    }
    double lii = sqrt(aii - ldotl);
    L.data[col_idx_i] = lii;
    //printf("lii return\n");
    return lii;
}

double get_lji(Mat& A, Mat& L, int i, int j, double lii)
{
    //if((i>=A.n_cols) || (j>=A.n_rows)){
    //printf("lji i:%d j:%d\n",i,j);}
    double lidotlj = sparse_row_partial_inner_product(L,i,j,i-1);
    //printf("srpip \n");
    double aij;
    for(int k = A.rowptr[i+1] -1; (k>=A.rowptr[i]) && (A.col_idx[k]>=j); --k){ aij = A.data[k];}
    //printf("xrpip \n");
    return (aij - lidotlj) / lii;
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
    for(int j = 1; j < l.n_rows; ++j)
    {
        int start, end;
        start = A.rowptr[j];
        end = A.rowptr[j+1];
    
        double lii = get_lii(A,l,j-1);
        for(int k = start; k < end; ++k)
        {

            if(A.col_idx[k]<j){
                
                l.data[k] = get_lji(A,l,A.col_idx[k],j,lii);
            }
            else if(A.col_idx[k] > j){
                l.data[k] = 0;
            }  
        }
        
    }
    get_lii(A,l,l.n_rows -1);
    //printf("safe  %d\n", A.n_rows);

    printf("cholesky had %d values that differed by more than tolerance\n",test_cholesky(A,l,1e-5));
}

void incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    incomplete_cholesky(A.on_proc, x, b);
}

