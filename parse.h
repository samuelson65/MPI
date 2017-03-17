#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <string>
#include <iostream>

#define MAXFLDS 40
#define ROWS 500 //in the file we have 501 rows. But we ignore the first row. See the parse function.
#define COLS 117 // We use zero indexing for both row and columns
#define NUM_CITIES 500
#define FILENAME "500_Cities__City-level_Data__GIS_Friendly_Format_.csv"

/*
  methods
*/
char *** parse(std::string filename);
void cleanup(char***);
void viewDataRow(int, char***);
int convert_string_to_int_index(std::string c);


char *** parse(std::string filename){
  std::ifstream read;
  read.open(filename);

  //create a matrix of c_strings
  char ***map = (char ***) malloc(ROWS * sizeof(char **));

  //For fields with values enclosed with parenthesis, we mark it with "NA"
  std::string na = "NA";
  std::string line;
  char *token;

  //ignore the first line
  std::getline(read, line);

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

int convert_string_to_int_index(std::string s){
  //we have -1 at the end because we are using zero indexing
  if(s.size() > 1)
    return ((int)s[0] - 64)*26 + (int)s[1] -64 - 1;
  else
    return (int)(s[0])- 64 -1;
}

void viewDataRow(int rownum, char*** map){
  for(int i = 0; i < COLS; i++)
    std::cout << map[rownum][i] << " ";
  std::cout << "\n";
}
