#include <mpi.h>
#include "parse.h"

static void process(int rank, int communicatorSize, std::string *cmd);
static void do_rank_0_work(int citiesToCompute, std::string *cmd);
static void do_rank_i_work(int citiesToCompute, std::string *cmd);

//void maxloc_val_rank_position

typedef struct pass_in_data_struc{
  double val;
  int rank;
} pass_in_data;



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
      std::cout << "This is column:" << convert_string_to_int_index(cmd[3]) << "\n";
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
    //exp
    /*
    MPI_Datatype mpi_val_rank_pos;
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_INT, MPI_INT }; //value, rank, position
    MPI_Aint disps[3] = { offsetof(mpi_val_rank_pos, val),
                    offsetof(mpi_val_rank_pos, rank),
                    offsetof(mpi_val_rank_pos, pos),  }
    int lens[3] = {1,0,0}; //data we need to define for MPI to create a type for
    MPI_Type_create_struct(3, lens, disps, types, &mpi_index_value);
    MPI_Type_commit(&mpi_dbl_twoindex);
    */
    //end exp
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
  int column_index = convert_string_to_int_index(cmd[3]);

  //preparing the data to send
  pass_in_data *total_pass_in = new pass_in_data[ROWS];
  for(int i = 0; i < ROWS; i++){
    total_pass_in[i].val = (double)atof(data[i][column_index]);
    //std::cout << total_rows_of_data_at_that_column[i] << " ";
  }

  double *total_rows_of_data_at_that_column = new double[ROWS];
  for(int i = 0; i < ROWS; i++){
    total_rows_of_data_at_that_column[i] = (double)atof(data[i][column_index]);
    //std::cout << total_rows_of_data_at_that_column[i] << " ";
  }

    //preparing data 2nd version
  //val_rank_pos *data_at_that_column = new val_rank_pos[ROWS];


  if(cmd[1] == "sr"){
    if(cmd[2] == "max"){
      //double* rank0_rows = new double[citiesToCompute];
      pass_in_data* rank_0_proc = new pass_in_data[citiesToCompute];

      MPI_Scatter(total_pass_in, citiesToCompute, MPI_DOUBLE_INT,
                   rank_0_proc, citiesToCompute, MPI_DOUBLE_INT, // recvCount & type are ignored
                   0, MPI_COMM_WORLD);

      std::cout << "Distributing data to other processes..." << std::endl;

      //std::cout << "Print out the first 25 elements and find the max\n";
      /*
      double max = 0;
      for (int i = 0; i < ROWS; i++){
        //std::cout << total_rows_of_data_at_that_column[i] << " ";
        max = total_rows_of_data_at_that_column[i] > max ? total_rows_of_data_at_that_column[i] : max;
      }
      std::cout << "\nDone!\n";
      std::cout << "The max is: " << max << "\n";
      */
      // each process has an array of num(citiesToCompute) double

      for(int i = 0; i < citiesToCompute; i++)
        rank_0_proc[i].rank = 0;
      pass_in_data* ans = new pass_in_data[citiesToCompute];
      //double *result = new double[citiesToCompute];
      MPI_Reduce(rank_0_proc, ans, citiesToCompute,
        MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        double max_mpi = 0;
        int max_mpis_rank = 0;
        for(int i = 0; i < citiesToCompute; i++){
          std::cout << "Value: " << ans[i].val <<
          " Rank: " << ans[i].rank << "\n";
          if(ans[i].val > max_mpi){
            max_mpi = ans[i].val ;
            max_mpis_rank = ans[i].rank;
          }
        }

        std::cout << "\n\nMax value: " << max_mpi << "in rank: " << max_mpis_rank << "\n";
      /*
      double max_mpi = 0;
      for(int i = 0; i < citiesToCompute; i++){
        std::cout << result[i] << " ";
        max_mpi = result[i] > max_mpi? result[i] : max_mpi;
      }
            std::cout << "\nThe max from mpi is: " << max_mpi << "\n";
      */
    }
  }

  /*

  */


  delete[] total_rows_of_data_at_that_column;
  cleanup(data);
}

static void do_rank_i_work(int citiesToCompute, std::string *cmd){
      //double* my_rows_of_data_at_that_column = new double[citiesToCompute];
      pass_in_data* my_rows = new pass_in_data[citiesToCompute];
      MPI_Request dataReq;

      MPI_Scatter(NULL, 0, MPI_DOUBLE_INT, // sendBuf, sendCount, sendType ignored
    		my_rows, citiesToCompute, MPI_DOUBLE_INT,
    		0, MPI_COMM_WORLD); // rank "0" originated the scatter

      int i_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
      for(int i = 0; i < citiesToCompute; i++)
        my_rows[i].rank = i_rank;

      /*
      double max = 0;
      for(int i = 0; i < citiesToCompute; i++){
        max = my_rows_of_data_at_that_column[i] > max? my_rows_of_data_at_that_column[i] : max;
      }*/
      //std::cout << "max here at rank: " << max << "\n";

      MPI_Reduce(my_rows, 0, citiesToCompute,
        MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

      //delete [] my_rows_of_data_at_that_column;
}
