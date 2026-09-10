#ifndef MAT_OPS_HPP
#define MAT_OPS_HPP
double inner_product(std::vector<double> a, std::vector<double> b);
double inner_product(std::vector<double> a, std::vector<double>b, int is, int ie);
double sparse_row_partial_inner_product(Mat& A, int rowa,int rowb,int colend );
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y, int is, int ie);
void scale(double alpha, std::vector<double>& x);
void scale(double alpha, std::vector<double>& x, int is, int ie);

#endif //MAT_OPS_HPP
