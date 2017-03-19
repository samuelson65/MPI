#include "pass_in_data_struct.h"

void process(int rank, int communicatorSize, std::string *cmd);

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

    if ((NUM_CITIES % communicatorSize) != 0){
  		if (rank == 0)
  			std::cerr << "communicatorSize " << communicatorSize
  			          << " does not evenly divide number of cities = " << NUM_CITIES << '\n';
    }
    else
      process(rank, communicatorSize, cmd);
  }
  MPI_Finalize();
  return 0;
}

void process(int rank, int communicatorSize, std::string *cmd){

    int citiesToCompute = NUM_CITIES / communicatorSize;
    //std::cout << "\n\n citiesToCompute " << citiesToCompute << "\n\n";
    MPI_Datatype MPI_pass_in_data;
    defineStruct_pass_in_data(&MPI_pass_in_data);
    MPI_Op mpi_max_location_index;
    define_op_max_pass_in_data(&mpi_max_location_index);
    MPI_Op mpi_min_location_index;
    define_op_min_pass_in_data(&mpi_min_location_index);
    MPI_Op mpi_take_avg;
    define_op_avg_pass_in_data(&mpi_take_avg);

    if(rank == 0){
      char *** data = parse(FILENAME);
      char ** categories = parse_first_line(FILENAME);
      int column_index = convert_string_to_int_index(cmd[3]);
      std::string that_column_name = categories[column_index];
      pass_in_data *total_pass_in = new pass_in_data[ROWS];
      //double *total_double_pas_in = new double[ROWS];
      copy_data_at_that_column(data, ROWS, column_index, citiesToCompute, total_pass_in);
      //copy_value_at_that_column(data, ROWS, column_index, total_double_pas_in);

      if(cmd[1] == "sr"){
        //pass_in_data* rank_0_proc = new pass_in_data[citiesToCompute]; //don't need this
        pass_in_data* ans = new pass_in_data[citiesToCompute];
        //double* double_ans = new double[citiesToCompute];

        MPI_Scatter(total_pass_in, citiesToCompute, MPI_pass_in_data,
                     MPI_IN_PLACE, 0, MPI_pass_in_data,
                     0, MPI_COMM_WORLD);
        if(cmd[2] == "max"){
          MPI_Reduce(total_pass_in, ans, citiesToCompute,
          MPI_pass_in_data, mpi_max_location_index, 0, MPI_COMM_WORLD);
          report_maxmin_answer(that_column_name, data, ans, citiesToCompute, "max");  //get the row index
        } else if(cmd[2] == "min") {
          MPI_Reduce(total_pass_in, ans, citiesToCompute,
          MPI_pass_in_data, mpi_min_location_index, 0, MPI_COMM_WORLD);
          report_maxmin_answer(that_column_name, data, ans, citiesToCompute, "min");
        } else if(cmd[2] == "avg") {
          MPI_Reduce(total_pass_in, ans, citiesToCompute,
          MPI_pass_in_data, mpi_take_avg, 0, MPI_COMM_WORLD);
          //exp
          double sum=0, avg = 0;
          for(int i= 0; i < ROWS; i++){
            sum += total_pass_in[i].val;
          }
          std::cout << "Correct avg: " << sum/ROWS << "\n";
          sum=0;
          for(int i = 0; i < citiesToCompute; i++){
            //std::cout << "SUM AT EACH PARTITION: " << ans[i].val << "\n";
            sum+=ans[i].val;
          }
          avg = sum/ROWS;
          std::cout << "Computed avg: " << avg << "\n";
          //end exp


        }
        delete[] ans;
      }
      cleanup(data, categories);
    }
    else{ //if rank != 0
      pass_in_data* my_rows = new pass_in_data[citiesToCompute];
      MPI_Request dataReq;

      MPI_Scatter(NULL, 0, MPI_pass_in_data, // sendBuf, sendCount, sendType ignored
        my_rows, citiesToCompute, MPI_pass_in_data,
        0, MPI_COMM_WORLD); // rank "0" originated the scatter
      if(cmd[2] == "max"){
        MPI_Reduce(my_rows, 0, citiesToCompute,
        MPI_pass_in_data, mpi_max_location_index, 0, MPI_COMM_WORLD);
      } else if(cmd[2] == "min") {
        MPI_Reduce(my_rows, 0, citiesToCompute,
        MPI_pass_in_data, mpi_min_location_index, 0, MPI_COMM_WORLD);
      } else if(cmd[2] == "avg") {
        MPI_Reduce(my_rows, 0, citiesToCompute,
        MPI_pass_in_data, mpi_take_avg, 0, MPI_COMM_WORLD);
      }
    }
}
