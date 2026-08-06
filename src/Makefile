default:cg pcg
	echo ":)"

cg: 
	mpicxx cg.cpp -o cg
pcg: 
	mpicxx spmv.cpp pcg.cpp -v -o pcg
clean:
	rm cg; rm pcg
