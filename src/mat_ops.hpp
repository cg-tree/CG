#ifndef MAT_OPS_HPP
#define MAT_OPS_HPP
#include "comm/comm_routines.hpp"
#include <functional>

enum INNER_PRODUCT_TYPE{
    STANDARD_INNER_PRODUCT,
    NON_BLOCKING_INNER_PRODUCT
};

std::function<void(std::vector<double>&,
    std::vector<double>&,
    double*,
    double*,
    MPI_Request*)> get_inner_product(INNER_PRODUCT_TYPE inner_product_type);
    
std::function<void(std::vector<double>&,
    std::vector<double>&,
    std::vector<double>&,
    double*,
    double*,
    MPI_Request*)> get_combined_inner_product(INNER_PRODUCT_TYPE inner_product_type);

std::function<double(std::vector<double>&, std::vector<double>&, AllreduceComm*)> get_inner_product_advance();

std::function<void(MPI_Request*, MPI_Status*)> get_inner_product_wait(INNER_PRODUCT_TYPE inner_product_type);

double inner_product(std::vector<double>& a,
                     std::vector<double>& b,
                     AllreduceComm* comm);
double inner_product(std::vector<double>& a, std::vector<double>& b);
double inner_product(std::vector<double>& a, std::vector<double>& b, int is, int ie);
double sparse_row_partial_inner_product(Mat& A, int rowa,int rowb,int colend );
double sparse_row_vector_inner_product(Mat& A, int rowa, std::vector<double>& x);

void Iinner_product(std::vector<double>& a, std::vector<double>& b, double* sum_local, double* sum, MPI_Request* request);
void combined_Iinner_product(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,double* sum_local, double* sum, MPI_Request* request);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y, int is, int ie);
void scale(double alpha, std::vector<double>& x);
void scale(double alpha, std::vector<double>& x, int is, int ie);

void forward_solve(Mat& L, std::vector<double>& y, std::vector<double>& b);
void backward_solve(Mat& L, std::vector<double>& x, std::vector<double>& b);
#endif //MAT_OPS_HPP
