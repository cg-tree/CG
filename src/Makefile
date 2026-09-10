default:cg pcg
	echo ":)"

cg: 
	mpicxx cg.cpp\
		spmv.cpp\
		mat_ops.cpp\
		icf.cpp\
		-o cg
pcg: 
	mpicxx spmv.cpp pcg.cpp -v -o pcg
clean:
	rm cg; rm pcg
