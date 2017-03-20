#include "pass_in_data_struct.h"
#include <map>

void process(int rank, int communicatorSize, std::string *cmd, int argc);

MPI_Datatype MPI_pass_in_data;
MPI_Op mpi_max_location_index;
MPI_Op mpi_min_location_index;
MPI_Op mpi_take_sum;
MPI_Op mpi_num_gtlt;

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
    // cmd[3] - city-column: the column index of the city with respect to the input; "A" is 0, "AA" is 26, "DM" is 116, etc.
    // cmd[4] - condition for sr-number method(gt,lt)  || city-column for bd method
    // cmd[5] - a numeric value for sr-number-condition method || city-column for bd method
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

    if (cmd[1] == "sr" && (NUM_CITIES % communicatorSize) != 0){
  		if (rank == 0) std::cerr << "communicatorSize " << communicatorSize << " does not evenly divide number of cities = " << NUM_CITIES << '\n';}
    else if(cmd[1] == "bg" && (communicatorSize != argc - 3)){
      if (rank == 0) std::cerr << "communicatorSize " << communicatorSize << " does not equal to the process about to execute for BG method\n";}
    else
      process(rank, communicatorSize, cmd, argc);
  }
  MPI_Finalize();
  return 0;
}

void process(int rank, int communicatorSize, std::string *cmd, int argc){

    int citiesToCompute = NUM_CITIES / communicatorSize;
    defineStruct_pass_in_data(&MPI_pass_in_data);
    define_op_max_pass_in_data(&mpi_max_location_index);
    define_op_min_pass_in_data(&mpi_min_location_index);
    define_op_sum_pass_in_data(&mpi_take_sum);
    define_op_numGtLt_pass_in_data(&mpi_num_gtlt);

    //map the MPI operations
    std::map<std::string, MPI_Op> MPI_operation_map = {{"max", mpi_max_location_index}, {"min", mpi_min_location_index},
    {"avg", mpi_take_sum}, {"number", mpi_num_gtlt}};

    if(rank == 0){
      //array of function pointers
      void (*report_answer[])(std::string*, char***, pass_in_data*, int, std::string* , std::string* ) =
      {report_maxmin_answer, report_avg_answer, report_gtlt_answer};
      std::map<std::string, int> report = {{"max", 0}, {"min", 0}, {"avg", 1}, {"number", 2}};

      char *** data = parse(FILENAME);
      char ** categories = parse_first_line(FILENAME);
      if(cmd[1] == "sr"){
        int column_index = convert_string_to_int_index(cmd[3]);
        std::string that_column_name = categories[column_index];
        pass_in_data *total_pass_in = new pass_in_data[ROWS];
        copy_data_at_that_column(data, ROWS, column_index, citiesToCompute, total_pass_in, 0);

        pass_in_data* ans = new pass_in_data[citiesToCompute];
        if(cmd[2] == "number" && argc >= 6){
          number_to_compare = std::stod(cmd[5]);
          //std::cout << "NUM TO compare " << number_to_compare << "\n";
          if(cmd[4] == "gt") adjust_number_gt(ROWS, total_pass_in);
          if(cmd[4] == "lt") adjust_number_lt(ROWS, total_pass_in);
        }

        MPI_Scatter(total_pass_in, citiesToCompute, MPI_pass_in_data, MPI_IN_PLACE, 0, MPI_pass_in_data, 0, MPI_COMM_WORLD);
        MPI_Reduce(total_pass_in, ans, citiesToCompute, MPI_pass_in_data, MPI_operation_map[cmd[2]], 0, MPI_COMM_WORLD);
        report_answer[report[cmd[2]]](&that_column_name, data, ans, citiesToCompute, &cmd[2], &cmd[4]);

        delete[] ans;
        cleanup(data, categories);
      }

      if(cmd[1]== "bg"){
        int num_columns = argc-3;
        int data_chunk = ROWS*num_columns;
        int process_col_index[num_columns];
        std::cout << "ROWS*num_columns: " << data_chunk << "\n";
        pass_in_data *total_pass_in = new pass_in_data[data_chunk];
        for(int i = 0; i < num_columns; i++){
          process_col_index[i] = convert_string_to_int_index(cmd[3 + i]);
          copy_data_at_that_column(data, ROWS, process_col_index[i], ROWS, total_pass_in, i*ROWS);
        }

        double ans[num_columns];
        ans[0] = bg_method(total_pass_in, 0, cmd[2]); // 0, since this is rank 0
        MPI_Bcast(total_pass_in, data_chunk, MPI_pass_in_data, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, &ans, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for(int i = 0; i < num_columns; i++)
          std::cout << ans[i] << "\n";
      }
    }
    else{ //if rank != 0
      if(cmd[1] == "sr"){
        pass_in_data* my_rows = new pass_in_data[citiesToCompute];
        MPI_Scatter(NULL, 0, MPI_pass_in_data, my_rows, citiesToCompute, MPI_pass_in_data, 0, MPI_COMM_WORLD); // rank "0" originated the scatter
        MPI_Reduce(my_rows, 0, citiesToCompute, MPI_pass_in_data, MPI_operation_map[cmd[2]], 0, MPI_COMM_WORLD);
      }
      if(cmd[1] == "bg"){
        pass_in_data* my_cpy = new pass_in_data[(argc-3)*ROWS];
        MPI_Bcast(my_cpy, (argc-3)*ROWS, MPI_pass_in_data, 0, MPI_COMM_WORLD);

        int i_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
        int start = i_rank*ROWS;
      //  std::cout << "I'm rank:" << i_rank << " my start: " << start << " my_end: " <<  ROWS+start << "\n";
        double answer = bg_method(my_cpy, i_rank, cmd[2]);
        MPI_Gather(&answer, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);


      }
    }//end big else block




}// end process
