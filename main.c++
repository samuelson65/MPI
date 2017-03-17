#include <mpi.h>
#include "parse.h"

static void process(int rank, int communicatorSize, std::string *cmd);
static void do_rank_0_work(int citiesToCompute, std::string *cmd);
static void do_rank_i_work(int citiesToCompute, std::string *cmd);

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

  if (argc < 4){
    if (rank == 0) // only report error once
      std::cerr << "Specify method, function outcome, and specified column on the command line.\n";
  }
  else{
    std::string cmd[argc];
    for(int i = 0; i < argc; i++)
      cmd[i] = argv[i];
    // cmd[0] -
    // cmd[1] - type of method: sr or bd
    // cmd[2] - function outcomes: max, min, avg, or conditional quantities
    // cmd[3] - the column index of the city with respect to the input; "A" is 0, "AA" is 26, "DM" is 116, etc.
    // cmd[4] - condition for sr-number method         || city-column for bd method
    // cmd[5] - number for  sr-number-condition method || city-column for bd method
    // cmd[6] - city-column for bd method
    // cmd[7] - city-column for bd method, and so on..

    if(rank == 0){
      for(int i = 0; i < argc; i++)
        std::cout << "This is argument " << i << " " << argv[i] << "\n";
  		std::cout << "communicatorSize: " << communicatorSize << "\n";
      std::cout << "This is command: " << cmd[1] << "\n";
      //viewDataRow(0,data);
    }
    process(rank, communicatorSize, cmd);
  }
  MPI_Finalize();
  return 0;
}

static void process(int rank, int communicatorSize, std::string *cmd){
  if ((NUM_CITIES % communicatorSize) != 0){
		if (rank == 0)
			std::cerr << "communicatorSize " << communicatorSize
			          << " does not evenly divide number of cities = " << NUM_CITIES << '\n';
	} else {
    int citiesToCompute = NUM_CITIES / communicatorSize;
    //std::cout << "\n\n citiesToCompute " << citiesToCompute << "\n\n";

    if(rank == 0)
      do_rank_0_work(citiesToCompute, cmd);
    else
      do_rank_i_work(citiesToCompute, cmd);
  }
}

static void do_rank_0_work(int citiesToCompute, std::string *cmd){
  char *** data = parse(FILENAME);
  MPI_Request sendReq;
  std::cout << "Distributing data to other processes..." << std::endl;
  /*
  MPI_Iscatter(A, nRowsToCompute * P, MPI_DOUBLE,
               MPI_IN_PLACE,       0, MPI_DOUBLE, // recvCount & type are ignored
               0, MPI_COMM_WORLD, &sendReq);
  */
  cleanup(data);
}

static void do_rank_i_work(int citiesToCompute, std::string *cmd){

}
