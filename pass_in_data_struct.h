#include <mpi.h>
#include "parse.h"


static double number_to_compare = 0.0; // number used for compare methods (gt and lt)
std::string no_num_error_message = "Oops looks like you did not enter a number to compare\n";
// "number blocks" in the struct
static const int NUM_STRUCT_FIELDS = 4;
void defineStruct_pass_in_data(MPI_Datatype* typeToBeCreated);
void define_op_max_pass_in_data(MPI_Op* op);
void max_location_index(void *in, void *inout, int *len, MPI_Datatype *type);
/*
  structure for MPI to pass in one column of data
*/
typedef struct pass_in_data_struc{
  double val;
  int rank;
  int index;
  int number;
} pass_in_data;

// template for MPI opertations
void max_location_index(void *in, void *inout, int *len, MPI_Datatype *type){
    pass_in_data *invals    = (pass_in_data*)in;
    pass_in_data *inoutvals = (pass_in_data*)inout;

    //remember this is global operations?
    //std::cout << "LEN: " <<*len<<"\n";
    for (int i=0; i<*len; i++) {
        if (invals[i].val > inoutvals[i].val) {
            inoutvals[i].val  = invals[i].val;
            //inoutvals[i].rank = invals[i].rank;
            inoutvals[i].index = invals[i].index;
        }
    }
    return;
}

// template for MPI opertations
void min_location_index(void *in, void *inout, int *len, MPI_Datatype *type){
    pass_in_data *invals    = (pass_in_data*)in;
    pass_in_data *inoutvals = (pass_in_data*)inout;
    for (int i=0; i<*len; i++) {
        if (invals[i].val < inoutvals[i].val) {
            inoutvals[i].val  = invals[i].val;
            //inoutvals[i].rank = invals[i].rank;
            inoutvals[i].index = invals[i].index;
        }
    }
    //std::cout << "Rank: " << inoutvals[i].rank << " Index: "<< inoutvals[24].index << " Value: " << inoutvals[24].val << "\n";
    return;
}

// template for MPI opertations
void take_sum(void *in, void *inout, int *len, MPI_Datatype *type){
    pass_in_data *invals    = (pass_in_data*)in;
    pass_in_data *inoutvals = (pass_in_data*)inout;
    //double sum = 0;

    /*
    int i_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    */
    for (int i=0; i<*len; i++)
        inoutvals[i].val += invals[i].val;
    /*
    for (int i=0; i<*len; i++)
      std::cout << "At rank " << i_rank << " and inoutvals[" << i <<"].val is " << inoutvals[i].val << "\n";
    */
    return;
}

void num_gt(void *in, void *inout, int *len, MPI_Datatype *type){
  pass_in_data *invals    = (pass_in_data*)in;
  pass_in_data *inoutvals = (pass_in_data*)inout;
  /*
  int i_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
  */
  for (int i=0; i<*len; i++)
      inoutvals[i].number += invals[i].number;
  /*
  for (int i=0; i<*len; i++) {
      double v1 = invals[i].val;
      double v2 = inoutvals[i].val;
      std::cout << "At rank " << i_rank << " and v1 " << v1 << " and v2 " << v2 << "\n";
      if(v1 > number_to_compare){
        //std::cout << "At rank " << i_rank << " and at" << i << " and Value is " << invals[i].val << "\n";
          inoutvals[i].number++;
      }
      if(v2 > number_to_compare){
        inoutvals[i].number++;
      }
  }
  */
  //for (int i=0; i<*len; i++)
  //  std::cout << "At rank " << i_rank << " and inoutvals[" << i <<"].number is " << inoutvals[i].number << "\n";
}

void num_lt(void *in, void *inout, int *len, MPI_Datatype *type){
  pass_in_data *invals    = (pass_in_data*)in;
  pass_in_data *inoutvals = (pass_in_data*)inout;
  for (int i=0; i<*len; i++)
      inoutvals[i].number += invals[i].number;
}


void copy_data_at_that_column(char*** data, int rows, int column_index, int partition_size, pass_in_data* p){
  for(int i = 0; i < rows; i++){
    p[i].val = (double)std::stod(data[i][column_index]);
    p[i].rank = (i+1)/partition_size; // since mpi scatter arrange each process in order => rank is the floor division: index / citiesToCompute
    p[i].index = i;
    p[i].number = 0;
  }
}

void adjust_number_gt(int rows, pass_in_data* p){
  for(int i = 0; i < rows; i++){
    if(p[i].val > number_to_compare){
      p[i].number = 1;
    }
  }
}

void adjust_number_lt(int rows, pass_in_data* p){
  for(int i = 0; i < rows; i++){
    if(p[i].val < number_to_compare){
      p[i].number = 1;
    }
  }
}
/*
void copy_value_at_that_column(char*** data, int rows, int column_index, double* p){
  for(int i = 0; i < rows; i++)
    p[i] = (double)std::stod(data[i][column_index]);
}
*/

void defineStruct_pass_in_data(MPI_Datatype* typeToBeCreated){
  /* create our new data type and its associated methods */
  MPI_Datatype types[NUM_STRUCT_FIELDS];
  MPI_Aint displ[NUM_STRUCT_FIELDS];
  int blklen[NUM_STRUCT_FIELDS];

	types[0] = MPI_DOUBLE; // type of val
	types[1] = MPI_INT;   // type of rank
	types[2] = MPI_INT; // base type of index
  types[3] = MPI_INT; // type of number

  //if it's an array, then it has length >= 1
	blklen[0] = 1; // length of val
	blklen[1] = 1; // length of rank
	blklen[2] = 1; // length of index
  blklen[3] = 1; // length of number

  displ[0] = 0;
  pass_in_data sample;
  MPI_Aint base; // base address of a pass_in_data instance
  MPI_Get_address(&sample.val, &base);
  // base address of successive fields of a given pass_in_data instance:

  /* FOR EDUCATION PURPOSES:
  int i_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);

  if(i_rank == 0){
    std::cout << "BASE " << base << "\n";
    std::cout << "disp[0] is at 0\n";
  }
  */
  MPI_Aint oneField;
  MPI_Get_address(&sample.rank, &oneField);
  /*
  if(i_rank == 0){
    std::cout << "ADDRESS one field " << oneField << "\n";
    std::cout << "disp[1] is on: " << oneField - base << "\n";
  }
  */
  displ[1] = oneField - base; // offset (displacement) to
  MPI_Get_address(&sample.index, &oneField);
  /*
  if(i_rank == 0){
    std::cout << "ADDRESS one field " << oneField << "\n";
    std::cout << "disp[2] is on: " << oneField - base << "\n";
  }
  */
  displ[2] = oneField - base; // offset (displacement) to
  MPI_Get_address(&sample.number, &oneField);
  /*
  if(i_rank == 0){
    std::cout << "ADDRESS one field " << oneField << "\n";
    std::cout << "disp[3] is on: " << oneField - base << "\n";
  }
  */
  displ[3] = oneField - base;

  MPI_Type_create_struct(NUM_STRUCT_FIELDS, blklen, displ, types, typeToBeCreated);
  MPI_Type_commit(typeToBeCreated);
}

void define_op_max_pass_in_data(MPI_Op* op){
  MPI_Op_create(max_location_index, 1, op);
}
void define_op_min_pass_in_data(MPI_Op* op){
  MPI_Op_create(min_location_index, 1, op);
}
void define_op_sum_pass_in_data(MPI_Op* op){
    MPI_Op_create(take_sum, 1, op);
}
void define_op_numGt_pass_in_data(MPI_Op* op){
  MPI_Op_create(num_gt, 1, op);
}
void define_op_numLt_pass_in_data(MPI_Op* op){
  MPI_Op_create(num_lt, 1, op);
}
/*
  in:
    std::string name - the name of the column
    char*** data - the data matrix; use this to print out the city and state's name
    pass_in_data* p - the answer (an array of pass_in_data_struct)
    int length - the length of pass_in_data
    std::string s - "max" or "min"
*/
void report_maxmin_answer(std::string name, char*** data, pass_in_data* p, int length, std::string s){
  double val_mpi = p[0].val;
  int val_mpi_rank = p[0].rank;
  int val_mpi_index = p[0].index;
  if(s == "max"){
    for(int i = 1; i < length; i++){
      //std::cout << "Value: " << p[i].val << " Rank: " << p[i].rank << "\n";
      if(p[i].val > val_mpi){
        val_mpi = p[i].val ;
        val_mpi_rank = p[i].rank;
        val_mpi_index = p[i].index;
      }
    }
    std::cout << data[val_mpi_index][1] << ", " <<  data[val_mpi_index][0] << ", " << name << " = " << (int)val_mpi << "\n";
  } else {
    for(int i = 1; i < length; i++){
      //std::cout << "Value: " << p[i].val << " Rank: " << p[i].rank << "\n";
      if(p[i].val < val_mpi){
        val_mpi = p[i].val ;
        val_mpi_rank = p[i].rank;
        val_mpi_index = p[i].index;
      }
    }
    std::cout << data[val_mpi_index][1] << ", " <<  data[val_mpi_index][0] << ", " << name << " = " << (int)val_mpi << "\n";
  }
  return;
}

void report_avg_answer(std::string name, pass_in_data* p, int length){
  double sum = 0, avg = 0;
  for(int i = 0; i < length; i++)
    sum+=p[i].val;
  avg = sum/ROWS;
  std::cout << "Average " << name << " = " << avg << std::endl;
}

void report_gtlt_answer(std::string name, pass_in_data* p, int length, std::string cmd){
  double sum = 0;
  for(int i = 0; i < length; i++)
    sum+=p[i].number;
  std::cout << "Number Cities with " << name << " " << cmd << " " <<number_to_compare << " = " << sum << "\n";
}



/*
void cleanUp_mpi(MPI_Datatype* typeToBeCreated){
  MPI_Op_free(&mpi_max_location_index);
  MPI_Type_free(&mpi_val_rank_pos);
}*/


// Below comment is from http://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node80.htm

/*
function is the user-defined function, which must have the following four arguments: invec, inoutvec, len and datatype.
The ANSI-C prototype for the function is the following.
typedef void MPI_User_function( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

The datatype argument is a handle to the data type that was passed into the call to MPI_REDUCE.
The user reduce function should be written such that the following holds: Let u[0], ... , u[len-1] be the len elements
in the communication buffer described by the arguments invec, len and datatype when the function is invoked;
let v[0], ... , v[len-1] be len elements in the communication buffer described by the arguments inoutvec,
len and datatype when the function is invoked; let w[0], ... , w[len-1] be len elements in the communication buffer
described by the arguments inoutvec, len and datatype when the function returns; then w[i] = u[i]°v[i],
for i=0 , ... , len-1, where ° is the reduce operation that the function computes.

Informally, we can think of invec and inoutvec as arrays of len elements that function is combining.
The result of the reduction over-writes values in inoutvec, hence the name.
Each invocation of the function results in the pointwise evaluation of the reduce operator on len elements:
I.e, the function returns in inoutvec[i] the value invec[i] ° inoutvec[i], for i=0, ... , count-1, where ° is the combining operation computed by the function.
*/


/*
MPI_OP_CREATE binds a user-defined global operation to an op handle that can subsequently be used in MPI_REDUCE, MPI_ALLREDUCE, MPI_REDUCE_SCATTER, and
MPI_SCAN. The user-defined operation is assumed to be associative. If commute = true, then the operation should be both commutative and associative.
If commute = false, then the order of operands is fixed and is defined to be in ascending, process rank order, beginning with process zero.
The order of evaluation can be changed, talking advantage of the associativity of the operation.
If commute = true then the order of evaluation can be changed, taking advantage of commutativity and associativity.
*/
