#ifndef SPMV_H
#define SPMV_H

double get_total_test_time();
int get_total_test_count();
// Serial SpMV b = alpha*A*x + eta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b);

// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b);

#endif //SPMV_H
