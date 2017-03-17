#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <string>
#include <iostream>
#include <mpi.h>

#define MAXFLDS 40
#define ROWS 501
#define NUM_CITIES 500
#define COLS 117
#define FILENAME "500_Cities__City-level_Data__GIS_Friendly_Format_.csv"

char *** parse(std::string filename);
void cleanup(char***);
void viewDataRow(int, char***);
static void process(int rank, int communicatorSize, std::string *cmd);
static void do_rank_0_work(int citiesToCompute, std::string *cmd);
static void do_rank_i_work(int citiesToCompute, std::string *cmd);

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);
	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
  std::string cmd = argv[1];

  if(rank == 0){
    for(int i = 0; i < argc; i++)
      std::cout << "This is argument " << i << " " << argv[i] << "\n";
		std::cout << "communicatorSize: " << communicatorSize << "\n";
    std::cout << "This is command: " << cmd << "\n";
    //viewDataRow(0,data);
  }
  process(rank, communicatorSize, &cmd);
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
      do_rank_0_work(citiesToCompute);
    else
      do_rank_i_work(citiesToCompute);
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



char *** parse(std::string filename){
  std::ifstream read;
  read.open(filename);

  //create a matrix of c_strings
  char ***map = (char ***) malloc(ROWS * sizeof(char **));

  //For fields with values enclosed with parenthesis, we mark it with "NA"
  std::string na = "NA";
  std::string line;
  char *token;

  //I refered this:
  //http://stackoverflow.com/questions/9244976/dynamically-allocate-a-string-matrix-in-c

  for (int i = 0; i < ROWS; ++i){
     map[i] = (char **) malloc(COLS * sizeof(char*));
     std::getline(read, line);
     token = strtok((char*)line.c_str(), ",");
     for (int j = 0; j < COLS; ++j){
        if(token[0] == '"'){
            token = strtok(NULL, "\"");
            strcpy(token, na.c_str());
        }
        map[i][j] = (char *) malloc(MAXFLDS * sizeof(char));
        strcpy(map[i][j], token);
        //std::cout << "[" << map[i][j] << "] ";
        token = strtok(NULL, ",");
     }
     //printf("\n");
  }

  return map;
}

void cleanup(char*** map){
  //clean up
  for (int i = 0; i < ROWS; ++i){
   for (int j = 0; j < COLS; ++j)
      free(map[i][j]);
   free(map[i]);
  }
  free(map);
}

void viewDataRow(int rownum, char*** map){
  for(int i = 0; i < COLS; i++)
    std::cout << map[rownum][i] << " ";
  std::cout << "\n";
}
