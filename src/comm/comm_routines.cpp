#include "comm_routines.hpp"


SpMVComm* get_spmv_comm(SPMV_COMM_TYPE spmv_comm_type){
    switch(spmv_comm_type){
        case STANDARD_SPMV_COMM: 
            return new SpMVComm();
        case PERSISTENT_SPMV_COMM:
            return (SpMVComm*)new SpMVCommPers();

    }
}


AllreduceComm* get_allreduce_comm(ALLREDUCE_COMM_TYPE allreduce_comm_type){
    switch(allreduce_comm_type){
        case STANDARD_ALLREDUCE_COMM: 
            return new AllreduceComm();
        case PERSISTENT_ALLREDUCE_COMM:
            return new AllreduceCommPers();

    }
}

