#include "pass_in_data_struct.h"
#include <map>
#include <memory>

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
  } else {
    std::string cmd[argc];
    for(int i = 0; i < argc; i++)
      cmd[i] = argv[i];
    // cmd[0] - ./proj2
    // cmd[1] - type of method: sr or bd
    // cmd[2] - function outcomes: max, min, avg, or conditional quantities
    // cmd[3] - city-column: the column index of the city with respect to the input; "A" is 0, "AA" is 26, "DM" is 116, etc.
    // cmd[4] - condition for sr-number method(gt,lt)  || city-column for bd method
    // cmd[5] - a numeric value for sr-number-condition method || city-column for bd method
    // cmd[6] - city-column for bd method
    // cmd[7] - city-column for bd method, and so on..
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
      void (*report_answer[])(std::string*, char***, pass_in_data*, int, std::string* , std::string* ) = {report_maxmin_answer, report_avg_answer, report_gtlt_answer};
      std::map<std::string, int> report = {{"max", 0}, {"min", 0}, {"avg", 1}, {"number", 2}};

      char *** data = parse(FILENAME);
      char ** categories = parse_first_line(FILENAME);

      if(cmd[1] == "sr"){ // argements after "proj2 sr" are just multiples of 2
        for(int i = 2; i < argc; i+= 2){
          int column_index = convert_string_to_int_index(cmd[i+1]);
          std::string that_column_name = categories[column_index];
          pass_in_data *total_pass_in = new pass_in_data[ROWS];
          copy_data_at_that_column(data, ROWS, column_index, citiesToCompute, total_pass_in, 0);
          if(cmd[i] == "number" && argc >= i+4){
            number_to_compare = std::stod(cmd[i+3]);
            if(cmd[i+2] == "gt") adjust_number_gt(ROWS, total_pass_in);
            if(cmd[i+2] == "lt") adjust_number_lt(ROWS, total_pass_in);
          }
          pass_in_data* ans = new pass_in_data[citiesToCompute];
          MPI_Scatter(total_pass_in, citiesToCompute, MPI_pass_in_data, MPI_IN_PLACE, 0, MPI_pass_in_data, 0, MPI_COMM_WORLD);
          MPI_Reduce(total_pass_in, ans, citiesToCompute, MPI_pass_in_data, MPI_operation_map[cmd[i]], 0, MPI_COMM_WORLD);
          report_answer[report[cmd[i]]](&that_column_name, data, ans, citiesToCompute, &cmd[i], &cmd[i+2]);
          MPI_Status sendStatus[2];
          delete[] ans;
          delete[] total_pass_in;
          i += cmd[i] == "number" ? 2 : 0;
        }//end for
        cleanup(data, categories);
      }

      if(cmd[1]== "bg"){
        int num_columns = argc-3;
        int data_chunk = ROWS*num_columns;
        int process_col_index[num_columns];
        pass_in_data *total_pass_in = new pass_in_data[data_chunk];
        for(int i = 0; i < num_columns; i++){
          process_col_index[i] = convert_string_to_int_index(cmd[3 + i]);
          copy_data_at_that_column(data, ROWS, process_col_index[i], ROWS, total_pass_in, i*ROWS);
        }
        double ans[num_columns];
        ans[0] = bg_method(total_pass_in, 0, cmd[2]); // 0, since this is rank 0
        MPI_Bcast(total_pass_in, data_chunk, MPI_pass_in_data, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, &ans, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        report_bg_answer(categories, ans, process_col_index, &cmd[2], num_columns);
        delete[] total_pass_in;
      }
    }
    else{ //if rank != 0
      if(cmd[1] == "sr"){
        for(int i = 2; i < argc; i+= 2){
          pass_in_data* my_rows = new pass_in_data[citiesToCompute];
          MPI_Scatter(NULL, 0, MPI_pass_in_data, my_rows, citiesToCompute, MPI_pass_in_data, 0, MPI_COMM_WORLD); // rank "0" originated the scatter
          MPI_Reduce(my_rows, 0, citiesToCompute, MPI_pass_in_data, MPI_operation_map[cmd[i]], 0, MPI_COMM_WORLD);
          i += cmd[i] == "number" ? 2 : 0;
          delete[] my_rows;
        }//end for
      }
      if(cmd[1] == "bg"){
        pass_in_data* my_cpy = new pass_in_data[(argc-3)*ROWS];
        MPI_Bcast(my_cpy, (argc-3)*ROWS, MPI_pass_in_data, 0, MPI_COMM_WORLD);
        int i_rank; MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
        int start = i_rank*ROWS;
        double answer = bg_method(my_cpy, i_rank, cmd[2]);
        MPI_Gather(&answer, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        delete[] my_cpy;
      }
    }//end else block
}// end process
