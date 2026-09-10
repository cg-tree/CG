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
            double aij_approx = sparse_row_partial_inner_product(L,i,j,A.n_cols);
            
            if( fabs(aij_approx - A.data[j]) > tol ){
                ++errorcount;
            }
        }
    }
    return errorcount;
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
    for(int j = start; A.col_idx[j] <= i; ++j)
    {
        aii = A.data[j];
        col_idx_i = j;
    }

    double ldotl = 0;
    if(i >0){
      ldotl = inner_product(L.data,L.data,start,col_idx_i);
    }
    double lii = sqrt(aii - ldotl);
    L.data[col_idx_i] = lii;
    return lii;
}

double get_lji(Mat& A, Mat& L, int i, int j, double lii)
{
    double lidotlj = sparse_row_partial_inner_product(L,i,j,i);
    double aij;
    for(int k = A.rowptr[i+1] -1; A.col_idx[k]>=j; --k){ aij = A.data[k];}
    return (aij - lidotlj) / lii;
}

void incomplete_cholesky(Mat& A, std::vector<double>& x, std::vector<double>& b)
{
    Mat* lptr = (Mat*)malloc(sizeof(Mat));
    Mat l = *lptr;
    l.rowptr = A.rowptr;
    l.col_idx = A.col_idx;
    l.data = A.data;
    l.n_rows = A.n_rows;
    l.n_cols= A.n_cols;
    l.nnz = A.nnz;

    for(int j = 1; j < l.n_rows; ++j)
    {
        int start, end;
        start = A.rowptr[j];
        end = A.rowptr[j+1];
    
        double lii = get_lii(A,l,j-1);
        for(int k = start; k < end; ++k)
        {

            if(A.col_idx[k]<j){
                l.data[k] = get_lji(A,l,k,j,lii);
            }
            else if(A.col_idx[k] == j){
                continue;
            }
            else{
                l.data[k] = 0;
            }  
        }
        
    }

    printf("cholesky had %d values that differed by more than tolerance\n",2* test_cholesky(A,l,1e-9));
    free(lptr);
    lptr = NULL;
}

void incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    incomplete_cholesky(A.on_proc, x, b);
}

