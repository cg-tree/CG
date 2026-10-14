default:cg benchmark_cg
	echo ":)"

cg: 
	mpicxx -O3 -std=c++11 \
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		benchmark_cg_mpi.cpp\
		-o cg
benchmark_cg: 
	mpicxx -O3 spmv.cpp mat_ops.cpp icf.cpp cg_routines.cpp benchmark_cg.cpp -o benchmark_cg
clean:
	rm cg; rm benchmark_cg
