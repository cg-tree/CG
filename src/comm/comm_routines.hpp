#ifndef COMM_ROUTINES_HPP
#define COMM_ROUTINES_HPP

#include "spmv_comm.hpp"
#include "spmv_comm_pers.hpp"
#include "allreduce_comm.hpp"
#include "allreduce_comm_pers.hpp"

enum SPMV_COMM_TYPE{
    STANDARD_SPMV_COMM,
    PERSISTENT_SPMV_COMM
};

SpMVComm* get_spmv_comm(SPMV_COMM_TYPE spmv_comm_type=STANDARD_SPMV_COMM);

enum ALLREDUCE_COMM_TYPE{
    STANDARD_ALLREDUCE_COMM,
    PERSISTENT_ALLREDUCE_COMM
};

AllreduceComm* get_allreduce_comm(ALLREDUCE_COMM_TYPE allreduce_comm_type=STANDARD_ALLREDUCE_COMM);

#ifdef MPIL
#include "spmv_comm_mpil.hpp"
#include "spmv_comm_mpil_pers.hpp"
#include "allreduce_comm_mpil.hpp"
#include "allreduce_comm_mpil_pers.hpp"
#endif

#endif
