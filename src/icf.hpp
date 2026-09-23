#ifndef ICF_HPP
#define ICF_HPP
void incomplete_cholesky(Mat& A, Mat& L);
void incomplete_cholesky(ParMat& A, Mat& L);
void incomplete_cholesky_solve( Mat& L, std::vector<double>& x, std::vector<double>& b);
void test_incomplete_cholesky(Mat& A, std::vector<double>& x, std::vector<double>& b);
void test_incomplete_cholesky(ParMat& A, std::vector<double>& x, std::vector<double>& b);
#endif //ICF_HPP
