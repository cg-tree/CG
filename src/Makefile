default:cg benchmark_cg
	echo ":)"

cg: 
	mpicxx -O3 -std=c++11 \
		spmv.cpp\
		comm/comm_routines.cpp\
		mat_ops.cpp\
		icf.cpp\
		benchmark_cg_mpi.cpp\
		-o cg
benchmark_cg: 
	mpicxx -O3 -fopenmp \
		spmv.cpp \
		comm/comm_routines.cpp \
		mat_ops.cpp \
		icf.cpp \
		cg_routines.cpp \
		benchmark_cg.cpp \
		-o benchmark_cg
mac_build_cg:#spack install mpich%apple-clang and brew install libomp 
	mpicxx -O3 -Xclang -fopenmp \
		spmv.cpp \
		comm/comm_routines.cpp \
		mat_ops.cpp \
		icf.cpp \
		cg_routines.cpp \
		benchmark_cg.cpp \
		-o benchmark_cg \
		-L/opt/homebrew/opt/libomp/lib -lomp
clean:
	rm cg; rm benchmark_cg
