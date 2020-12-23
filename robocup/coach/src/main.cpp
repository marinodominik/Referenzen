/*
 * Author: Manuel "Sputnick" Nickschas
 * started in 11/2001
 *
 * This is an experimental coach for use with the Karlsruhe Brainstormers BS01/02 agent.
 * 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "coach.h"
#include "logger.h"

//TG
#include <iostream>
#include <fstream>
std::ofstream gvDummyOutputStream("/dev/null");

int main(int argc,char **argv) {

  if(RUN::init(argc,argv)) {
    std::cout << "Coach        is started and ready to rumble!" << std::endl << std::flush;
    //  while(RUN::serverAlive) {
    RUN::mainLoop(argc,argv);
    //}	    
  
  }
  RUN::cleanUp();
  return 0;
}
