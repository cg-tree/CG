#ifndef SPMV_H
#define SPMV_H

#include "comm/comm_routines.hpp"
#include <functional>
double get_total_test_time();
double get_naive_total_test_time();
int get_total_test_count();
// Serial SpMV b = alpha*A*x + eta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b);

// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b);
enum SPMV_TYPE{
  STANDARD
};

std::function<void(double,ParMat&,std::vector<double>&,double,std::vector<double>&,SpMVComm*)> get_spmv(SPMV_TYPE SPMV);

#endif //SPMV_H
