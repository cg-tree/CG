default:cg pcg
	echo ":)"

cg: 
	mpicxx -O3 cg.cpp\
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		-o cg
pcg: 
	mpicxx -O3 spmv.cpp pcg.cpp -v -o pcg
clean:
	rm cg; rm pcg
