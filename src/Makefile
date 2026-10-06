default:cg pcg
	echo ":)"

cg: 
	mpicxx -O3 -std=c++11 \
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		benchmark_cg_mpi.cpp\
		-o cg
pcg: 
	mpicxx -O3 spmv.cpp pcg.cpp mat_ops.cpp icf.cpp -o pcg
clean:
	rm cg; rm pcg
