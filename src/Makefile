default:cg cg_cholesky pcg
	echo ":)"

cg: 
	mpicxx -O3 cg.cpp\
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		-o cg
cg_cholesky:
	mpicxx -O3 -DCHOLESKY cg.cpp\
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		-o cg_cholesky
pcg: 
	mpicxx -O3 spmv.cpp pcg.cpp -v -o pcg
clean:
	rm cg; rm cg_cholesky; rm pcg
