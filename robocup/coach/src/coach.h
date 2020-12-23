/* Author: Manuel "Sputnick" Nickschas, 11/2001
 *
 * This file declares main components of SputCoach, including most of the network
 * communication.
 *
 */

#ifndef _COACH_H_
#define _COACH_H_

#include "defs.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "udpsocket.h"
#include <stdlib.h>
#include <string.h>

extern Field fld;
extern bool onlineCoach;

namespace RUN {
  
    const int bufferMaxSize = UDPsocket::MAXMESG;
    extern UDPsocket sock;

    //const int side_NONE= 0;
    const int side_LEFT= 0;
    const int side_RIGHT= 1;
    extern int side;
    
    extern bool serverAlive;
    extern bool initDone;
    extern bool quit;

    bool init(int argc,char**argv);
    bool initPostConnect(int argc,char **argv);
    void cleanUp();
    long getCurrentMsTime(); // returns time in ms since first call to this routine
    void initOptions(int argc,char **argv);
    void initLogger(int argc,char **argv);
    bool initNetwork();
    void printPlayerTypes();
    void printPTCharacteristics();
    void announcePlayerChange(bool ownTeam,int unum,int type=-1);
    void mainLoop(int argc, char **argv);
}


#endif
