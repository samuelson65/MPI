#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <string>
#include <iostream>



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
