#ifndef MAT_OPS_HPP
#define MAT_OPS_HPP
double inner_product(std::vector<double> a, std::vector<double> b);
double inner_product(std::vector<double> a, std::vector<double>b, int is, int ie);
double sparse_row_partial_inner_product(Mat& A, int rowa,int rowb,int colend );
double sparse_row_vector_inner_product(Mat& A, int rowa, std::vector<double> x);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y, int is, int ie);
void scale(double alpha, std::vector<double>& x);
void scale(double alpha, std::vector<double>& x, int is, int ie);

void forward_solve(Mat& L, std::vector<double>& y, std::vector<double>& b);
void backward_solve(Mat& L, std::vector<double>& x, std::vector<double>& b);
#endif //MAT_OPS_HPP
