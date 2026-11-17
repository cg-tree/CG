#ifndef ALLREDUCE_COMM_HPP
#define ALLREDUCE_COMM_HPP

#include "mpi.h"
#include <functional>
#if defined(OPEN_MPI) && OMPI_MAJOR_VERSION < 5
  #include <mpi-ext.h>
  #define MPI_Allreduce_init MPIX_Allreduce_init
#endif
class AllreduceComm
{
    public:
        AllreduceComm()
        {
            local_sum = NULL;
            global_sum = NULL;
            count = 1;
            allreduce = MPI_Allreduce;
            wait_logic = [](){return;};
        }

        ~AllreduceComm()
        {

        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;
        }

        void start()
        {
             allreduce(local_sum, global_sum, count, MPI_DOUBLE, MPI_SUM, 
                  MPI_COMM_WORLD);
        }

        void wait()
        {
            wait_logic();
        }
        void set_count(int x){count = x;}
        int get_count(){return count;}
        void set_local_sum(double sum)
        {
            *local_sum = sum;
        }

        double get_global_sum()
        {
            return *global_sum;
        }
        
        void set_nonblocking(){
            using namespace std::placeholders;
            allreduce = std::bind(MPI_Iallreduce,_1,_2,_3,_4,_5,_6,&req);
            wait_logic = std::bind(MPI_Wait, &req, MPI_STATUS_IGNORE);
        }
        void set_blocking(){
            allreduce = MPI_Allreduce;
            wait_logic = [](){return;};
        }

    protected:
        double* local_sum;
        double* global_sum;
        int count;
    private:
        MPI_Request req;
        std::function<void(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)> allreduce;
        std::function<void()> wait_logic;
};

#endif

