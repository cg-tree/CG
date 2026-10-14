#ifndef CG_ROUTINES_HPP
#define CG_ROUTINES_HPP

enum CG_TYPE{
STANDARD_CG,
PRECONDITIONED_CG,
PIPE_CG};
std::function<int(ParMat&, 
        std::vector<double>&, 
        std::vector<double>&, 
        int,
        double)> get_cg( CG_TYPE cg_type, SPMV_TYPE spmv_type=STANDARD_SPMV, INNER_PRODUCT_TYPE inner_product_type=STANDARD_INNER_PRODUCT );
#endif //CG_ROUTINES_HPP
