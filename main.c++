#include "parse.h"
#include "pass_in_data_struct.h"



void process(int rank, int communicatorSize, std::string *cmd);
void do_rank_0_work(int citiesToCompute, std::string *cmd);
void do_rank_i_work(int citiesToCompute, std::string *cmd);


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

void process(int rank, int communicatorSize, std::string *cmd){
  if ((NUM_CITIES % communicatorSize) != 0){
		if (rank == 0)
			std::cerr << "communicatorSize " << communicatorSize
			          << " does not evenly divide number of cities = " << NUM_CITIES << '\n';
	} else {

    int citiesToCompute = NUM_CITIES / communicatorSize;
    //std::cout << "\n\n citiesToCompute " << citiesToCompute << "\n\n";
    MPI_Datatype MPI_pass_in_data;
    defineStruct_pass_in_data(&MPI_pass_in_data);
    MPI_Op mpi_max_location_index;
    define_op_max_pass_in_data(&mpi_max_location_index);

    if(rank == 0)
      do_rank_0_work(citiesToCompute, cmd);
    else
      do_rank_i_work(citiesToCompute, cmd);

  }
}

void do_rank_0_work(int citiesToCompute, std::string *cmd){

  MPI_Datatype MPI_pass_in_data;
  defineStruct_pass_in_data(&MPI_pass_in_data);
  MPI_Op mpi_max_location_index;
  define_op_max_pass_in_data(&mpi_max_location_index);

  char *** data = parse(FILENAME);
  MPI_Request sendReq;
  int column_index = convert_string_to_int_index(cmd[3]);

  //preparing the data to send
  pass_in_data *total_pass_in = new pass_in_data[ROWS];
  for(int i = 0; i < ROWS; i++){
    total_pass_in[i].val = (double)atof(data[i][column_index]);
    total_pass_in[i].rank = (i+1)/citiesToCompute; // since mpi scatter arrange each process in order => rank is the floor division: index / citiesToCompute
    total_pass_in[i].index = i+1;
    //std::cout << total_rows_of_data_at_that_column[i] << " ";
  }
  /*
  double *total_rows_of_data_at_that_column = new double[ROWS];
  for(int i = 0; i < ROWS; i++){
    total_rows_of_data_at_that_column[i] = (double)atof(data[i][column_index]);
    //std::cout << total_rows_of_data_at_that_column[i] << " ";
  }
  */


    //preparing data 2nd version
  //val_rank_pos *data_at_that_column = new val_rank_pos[ROWS];


  if(cmd[1] == "sr"){
    if(cmd[2] == "max"){
      //double* rank0_rows = new double[citiesToCompute];
      pass_in_data* rank_0_proc = new pass_in_data[citiesToCompute];

      MPI_Scatter(total_pass_in, citiesToCompute, MPI_pass_in_data,
                   rank_0_proc, citiesToCompute, MPI_pass_in_data, // recvCount & type are ignored
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
      /*
      for(int i = 0; i < citiesToCompute; i++)
        rank_0_proc[i].rank = 0;
      */
      pass_in_data* ans = new pass_in_data[citiesToCompute];
      //double *result = new double[citiesToCompute];



      MPI_Reduce(total_pass_in, ans, citiesToCompute,
        MPI_pass_in_data, mpi_max_location_index, 0, MPI_COMM_WORLD);

        double max_mpi = 0;
        int max_mpi_rank = 0;
        int max_mpi_index = 0;
        for(int i = 0; i < citiesToCompute; i++){
          std::cout << "Value: " << ans[i].val <<
          " Rank: " << ans[i].rank << "\n";
          if(ans[i].val > max_mpi){
            max_mpi = ans[i].val ;
            max_mpi_rank = ans[i].rank;
            max_mpi_index = ans[i].index;
          }
        }

        std::cout << "\n\nMax value: " << max_mpi << " in rank: " << max_mpi_rank <<  " in the fucking index:" << max_mpi_index << "\n";
        std::cout << "Table: " << *data[max_mpi_index] << "\n";

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


  //delete[] total_rows_of_data_at_that_column;
  cleanup(data);
}

void do_rank_i_work(int citiesToCompute, std::string *cmd){

      MPI_Datatype MPI_pass_in_data;
      defineStruct_pass_in_data(&MPI_pass_in_data);
      MPI_Op mpi_max_location_index;
      define_op_max_pass_in_data(&mpi_max_location_index);

      //double* my_rows_of_data_at_that_column = new double[citiesToCompute];
      pass_in_data* my_rows = new pass_in_data[citiesToCompute];
      MPI_Request dataReq;

      MPI_Scatter(NULL, 0, MPI_pass_in_data, // sendBuf, sendCount, sendType ignored
    		my_rows, citiesToCompute, MPI_pass_in_data,
    		0, MPI_COMM_WORLD); // rank "0" originated the scatter

      /*
      int i_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
      for(int i = 0; i < citiesToCompute; i++)
        my_rows[i].rank = i_rank;
      */
      /*
      double max = 0;
      for(int i = 0; i < citiesToCompute; i++){
        max = my_rows_of_data_at_that_column[i] > max? my_rows_of_data_at_that_column[i] : max;
      }*/
      //std::cout << "max here at rank: " << max << "\n";

      MPI_Reduce(my_rows, 0, citiesToCompute,
        MPI_pass_in_data, mpi_max_location_index, 0, MPI_COMM_WORLD);

      //delete [] my_rows_of_data_at_that_column;
}
