/* Author: Manuel "Sputnick" Nickschas, 11/2001
 * 
 * See coach.h for description.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "coach.h"
#include "param.h"
#include "field.h"
#include "logger.h"
#include "modules.h"
#include "messages.h"

#include "udpsocket.h"
#include "str2val.h"
#include "options.h"

//#define MAX_STRING 512

using namespace std;

Field fld;
bool onlineCoach;

bool RUN::quit=false;
bool RUN::serverAlive=false;
bool RUN::initDone=false;
int RUN::side;
UDPsocket RUN::sock;

/** Init coach */
bool RUN::init(int argc,char **argv) {
  getCurrentMsTime();
//  cout << "\n------------------------------------------------------------------";
  initOptions(argc,argv);
  initLogger(argc,argv);
  MSG::initMsg();
  if(!initNetwork()) {
    cout << "\nFATAL: Connection to server failed. Aborting...\n";
    return false;
  } else {
//    cout << "\nStarting message handling...";
    initDone=false;
  }
  return true;
}

/** Called when message loop is up and running */
bool RUN::initPostConnect(int argc,char **argv) {
  static int ourTeamCheck=0;
  static int hisTeamCheck=0;
  MSG::sendMsg(MSG::MSG_TEAM_NAMES);
  if(ourTeamCheck>=0 && Options::waitForOurTeam) {
//    if(!ourTeamCheck) cout << "\nWaiting for our team..." << flush;
    if(strlen(fld.getMyTeamName())==0) {
      ourTeamCheck=1;return false;
    }
    ourTeamCheck=-1;
    for(int i=0;i<TEAM_SIZE;i++) if(!fld.myTeam[i].alive) ourTeamCheck=1;
//    if(ourTeamCheck<0) cout << " "<<fld.getMyTeamName()<<" complete." << flush;
    else return false;
  }
  if(hisTeamCheck>=0 && Options::waitForHisTeam) {
//    if(!hisTeamCheck) cout << "\nWaiting for other team..." << flush;
    if(strlen(fld.getHisTeamName())==0) {
      hisTeamCheck=1;return false;
    }
    hisTeamCheck=-1;
    for(int i=0;i<TEAM_SIZE;i++) if(!fld.hisTeam[i].alive) hisTeamCheck=1;
//    if(hisTeamCheck<0) cout << " "<<fld.getHisTeamName()<<" complete." << flush;
    else 
    {
      if(strlen(fld.getHisTeamName())==0)
        return false;
      else
      {  
        LOG_FLD(0,<<"At time "<<fld.getTime()<<" initPostConnect returns true, "
          <<"although not all opponents have connected, yet. However, I already "
          <<"know his team name: "<<fld.getHisTeamName());  }
    }
  }
  
  MOD::loadModules();
  MOD::initModules(argc,argv);
//  cout  << "\nInitialisation complete.\n"
//	<< "------------------------------------------------------------------";
//
//  printPlayerTypes();
//  printPTCharacteristics();
//  if(Options::interactive) {
//    cout << "\nEntering interactive mode...\n";
//  } else {
//    cout << "\nEntering non-interactive mode...\n";
//  }
  initDone=true;
  return true;
}

void RUN::cleanUp() {
  MOD::destroyModules();
  MOD::unloadModules();
  LogOptions::close();
  RUN::sock.send_msg("(bye)",5);
  MSG::destructMsg();
  long t=RUN::getCurrentMsTime() / 1000;
  cout << "\nFRA-UNIted_Coach has been active for ";
  if (t/3600) cout << t/3600 << "h ";
  if (t/60)   cout << (t/60)%60   << "m ";
  cout << t % 60 << "s. Bye!\n\n";
}

/** Return time in ms since first call to this routine */
long RUN::getCurrentMsTime() {
    timeval tval;
    static long s_time_at_start= 0;
    if (gettimeofday(&tval,NULL))
      cerr << "\n something wrong with time measurement";

    if ( 0 == s_time_at_start ) 
      s_time_at_start= tval.tv_sec;

    return (tval.tv_sec - s_time_at_start) * 1000 + tval.tv_usec / 1000;
}

/** Read all options (command line and config file) */
void RUN::initOptions(int argc,char **argv) {
//  cout << "\nReading configuration data...";
  Options::read_config_name(argc,argv);
  Options::read_from_file(Options::coach_conf);
  Options::read_from_command_line(argc,argv);
  if(!isatty(0)) Options::interactive=0;      // kein stdin, falls im Background!
  onlineCoach = !Options::isTrainer;
  //Options::print_options();
}  

void RUN::initLogger(int argc,char **argv) {
  LogOptions::init();
  LogOptions::read_from_file(Options::coach_conf);
  LogOptions::read_from_command_line(argc,argv);

  char side_chr='l';
  if(RUN::side != RUN::side_LEFT) side_chr='r';

  sprintf(LogOptions::log_fname,"%s/%s-c-%c-actions.log",LogOptions::log_dir,
  Options::teamName,side_chr);

  //sprintf(LogOptions::log_fname,"%s/11-r-actions.log",LogOptions::log_dir);

//  cout << "\nStarting Logger...";
  if(LogOptions::use_file()) {
//    cout << "\nLogging to " << LogOptions::log_fname;
  }
  LogOptions::open();
}

/** Connect to server and send/receive initialisation messages */
bool RUN::initNetwork() 
{
  using namespace MSG;
  
  char buffer[bufferMaxSize];
  int num_bytes;

//  cout << "\nConnecting to server on " << Options::host << ":" << Options::port << "..." << flush;

  sock.init_socket_fd();

  sock.init_serv_addr(Options::host,Options::port);

  char ssVersion[10];
  sprintf(ssVersion,"%d",Options::cvServerProtocol);
  sendMsg(MSG_INIT,ssVersion,Options::teamName);

  //sock.recv_msg(buffer, num_bytes,true);

  bool dummy = recvAndParseMsg();
  if(dummy==MSG_INIT) 
  {

//    cout << "\nHandshaking with server complete (v="<<ssVersion<<"). Now processing param messages..."
//	 << "\nReceiving server params...             " << flush;
    ServerParam::init();
    if(recvAndParseMsg()==MSG_SERVER_PARAM) 
    {
//      cout << "OK.\nReceiving player params...             " << flush;
    } 
    else 
    {
      std::cerr << "ERROR!\nReceived message was:\n" << buffer;
      return false;
    }
    PlayerParam::init();
    if(recvAndParseMsg()==MSG_PLAYER_PARAM) 
    {
//      cout << "OK.\nReceiving heterogenous player types... " << flush;
    } 
    else 
    {
      std::cerr << "ERROR!\nReceived message was:\n" << buffer;
      return false;
    }
    for(int i=0;i<PlayerParam::player_types;i++) 
    {
      sock.recv_msg(buffer,num_bytes,true);
      fld.plType[i].init();
      if(!fld.plType[i].parseMsg(buffer)) 
      {
	    std::cerr << "ERROR!\nReceived message was:\n" << buffer;
    	return false;
      }
    }
//    cout << "OK." << flush;
    sock.set_fd_nonblock();
    return true;
  }
  cerr << "\nFATAL: Server did not send init message! "
       << "Received message was:\n" << msgBuffer << flush;
  return false;
}

void RUN::printPlayerTypes() {
  char buf[2048];
  cout << "\nPlayer Types are:"
       << "\n=================";

  int i;
  sprintf(buf,"\npt | player_speed_max | stamina_inc_max  |   player_decay   | inertia_moment  "
	  "\n------------------------------------------------------------------------------" );
  for(i=0;i<PlayerParam::player_types;i++) {
    sprintf(buf+strlen(buf),"\n%d  | %7.5f (%5.1f%%) | %7.4f (%5.1f%%) | %7.5f (%5.1f%%) "
	    "| %7.5f (%5.1f%%) ",i,
	    fld.plType[i].player_speed_max,
	    fld.plType[i].player_speed_max/fld.plType[0].player_speed_max*100,
	    fld.plType[i].stamina_inc_max,
	    fld.plType[i].stamina_inc_max/fld.plType[0].stamina_inc_max*100,
	    fld.plType[i].player_decay,
	    fld.plType[i].player_decay/fld.plType[0].player_decay*100,
	    fld.plType[i].inertia_moment,
	    fld.plType[i].inertia_moment/fld.plType[0].inertia_moment*100);
  }
  cout << buf;
  sprintf(buf,"\n\npt | dash_power_rate  |   player_size    | kickable_margin  |   kick_rand     "
	  "\n------------------------------------------------------------------------------" );
  for(i=0;i<PlayerParam::player_types;i++) {
    sprintf(buf+strlen(buf),"\n%d  | %7.5f (%5.1f%%) | %7.5f (%5.1f%%) | %7.5f (%5.1f%%) "
	    "| %7.5f (%5.1f%%) ",i,
	    fld.plType[i].dash_power_rate,
	    fld.plType[i].dash_power_rate/fld.plType[0].dash_power_rate*100,
	    fld.plType[i].player_size,
	    fld.plType[i].player_size/fld.plType[0].player_size*100,
	    fld.plType[i].kickable_margin,
	    fld.plType[i].kickable_margin/fld.plType[0].kickable_margin*100,
	    fld.plType[i].kick_rand,
	    fld.plType[i].kick_rand/fld.plType[0].kick_rand*100);
  }
  cout << buf;
  sprintf(buf,"\n\npt |  extra_stamina   |    effort_max    |    effort_min    |                 "
	  "\n------------------------------------------------------------------------------" );
  for(i=0;i<PlayerParam::player_types;i++) {
    sprintf(buf+strlen(buf),"\n%d  | %7.4f (%5.1f%%) | %7.5f (%5.1f%%) | %7.5f (%5.1f%%) |",i,
	    fld.plType[i].extra_stamina,
	    fld.plType[i].extra_stamina/fld.plType[0].extra_stamina*100,
	    fld.plType[i].effort_max,
	    fld.plType[i].effort_max/fld.plType[0].effort_max*100,
	    fld.plType[i].effort_min,
	    fld.plType[i].effort_min/fld.plType[0].effort_min*100);
  }
  cout << buf << "\n------------------------------------------------------------------------------\n"
       << flush;
  /*
    fld.plType[i].,
    fld.plType[i]./fld.plType[0].*100,
  */
}

void RUN::printPTCharacteristics() {
  std::ostringstream oss;
  oss << "\nPlayer Type characteristics:"
    //<< "\n----------------------------"
    ;
  for(int id=0;id<PlayerParam::player_types;id++) {    

      oss << "\n--------------------------------------------------------------"
	<< "\ntype " << id 
      //<< "\n dash_power_rate=          " << fld.plType[id].dash_power_rate
      //<< "\n player_decay=             " << fld.plType[id].player_decay
      //<< "\n kick_rand=                " << fld.plType[id].kick_rand
      //<< "\n kickable_area             " << fld.plType[id].player_size+fld.plType[id].kickable_margin
      //<< "\n--"
	<< "\n real_player_speed_max=    " << fld.plType[id].real_player_speed_max
	<< "\n dash_to_keep_max_speed=   " << fld.plType[id].dash_to_keep_max_speed
	<< "\n stamina_inc_max=          " << fld.plType[id].stamina_inc_max
      //<< "\n stamina_demand_per_cycle= " << stamina_demand_per_meter * real_player_speed_max 
	<< "\n stamina_demand_per_meter= " << fld.plType[id].stamina_demand_per_meter
	<< "\n--";
  oss.precision( 5 );
  oss
  << "\n speed_progression=        "<<endl
  << fld.plType[id].speed_progress[0] << "\t "
  << fld.plType[id].speed_progress[1] << "\t "
  << fld.plType[id].speed_progress[2] << "\t "
  << fld.plType[id].speed_progress[3] << "\t "
  << fld.plType[id].speed_progress[4] << "\t "
  << fld.plType[id].speed_progress[5] << "\t "
  << fld.plType[id].speed_progress[6] << "\t "
  << fld.plType[id].speed_progress[7] << "\t "
  << fld.plType[id].speed_progress[8]
  << "\n";
  oss
  << "   max_likelihood_max_speed_progression=        "<<endl
  << fld.plType[id].max_likelihood_max_speed_progress[0] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[1] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[2] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[3] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[4] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[5] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[6] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[7] << "\t "
  << fld.plType[id].max_likelihood_max_speed_progress[8]
  << "\n";
  oss
  << "   max_speed_progression=        "<<endl
  << fld.plType[id].max_speed_progress[0] << "\t "
  << fld.plType[id].max_speed_progress[1] << "\t "
  << fld.plType[id].max_speed_progress[2] << "\t "
  << fld.plType[id].max_speed_progress[3] << "\t "
  << fld.plType[id].max_speed_progress[4] << "\t "
  << fld.plType[id].max_speed_progress[5] << "\t "
  << fld.plType[id].max_speed_progress[6] << "\t "
  << fld.plType[id].max_speed_progress[7] << "\t "
  << fld.plType[id].max_speed_progress[8]
  << "\n";
    /*
    for (int i=1; i<; i++) {
      out << "(" << i << ": " << demand[i].dist << ", " << demand[i].stamina / demand[i].dist << ")"; 
      if ( i % 4 == 0)
	out << "\n";
    }
    */
    /*
      for (int i=1; i<PlayerType::max_demand; i++) {
      if ( fld.plType[id].stamina_demand[i-1].dist < 3.0 &&
      fld.plType[id].stamina_demand[i].dist >= 3.0  || 
      int(fld.plType[id].stamina_demand[i-1].dist / 5.0) <
      int(fld.plType[id].stamina_demand[i].dist / 5.0) )
      str << "\n(" << i << ": " << fld.plType[id].stamina_demand[i].dist
      << ", " << fld.plType[id].stamina_demand[i].stamina /
      fld.plType[id].stamina_demand[i].dist << ")"; 
      }
    */
    oss << "\navg stamina (10m) =     " << fld.plType[id].stamina_10m
	<< "\navg stamina (20m) =     " << fld.plType[id].stamina_20m
	<< "\navg stamina (30m) =     " << fld.plType[id].stamina_30m
	<< ends;
    cout << oss.str() << flush;
    oss.seekp(0);
  }
  //cerr << buf << "\n--------------------------------------------------------------" << flush;
  
}

void RUN::announcePlayerChange(bool ownTeam,int unum,int type) {
  if(ownTeam) {
    fld.myTeam[unum-1].type=type;
    fld.myTeam[unum-1].staminaCapacityBound = PLAYER_STAMINA_CAPACITY + ServerParam::stamina_max;
  } else {
    fld.hisTeam[unum-1].type=type;
  }
  for(int mod=0;mod<MOD::numModules;mod++) {
    MOD::coachModule[mod]->onChangePlayerType(ownTeam,unum,type);
  }
}

void RUN::mainLoop(int argc,char **argv) {
//cout<<"I am in the MAINLOOP!\n";
  fd_set rfds;
  struct timeval tv;
  int retval;
  int max_fd=  0;

  int msgType;
  int oldPM=0;
  static char buffer[bufferMaxSize];

  static int last_loop_time = -1;
  
  MSG::sendMsg(MSG::MSG_EYE,1);
  //MSG::sendMsg(MSG::MSG_EAR,1); //nicht erlaubt fuer Online-Coach!?
  while(serverAlive) 
  {
    
    max_fd=0;
    FD_ZERO(&rfds);
    if(Options::interactive) 
    {
      FD_SET(0,&rfds); //stdin
    }
    FD_SET(sock.get_socket_fd(),&rfds);
    if (sock.get_socket_fd() > max_fd) max_fd= sock.get_socket_fd();
    tv.tv_sec=5; tv.tv_usec=0;

    retval = select(max_fd + 1, &rfds, NULL, NULL, &tv);
    if(retval<=0) 
    {
      if(retval<0) 
      {
        cout << "\nSelect: Error occured" << flush;
        continue;
      }
      //retval == 0
      cout << "\nWaited for more than 5 seconds - shutting down!";
      serverAlive=false;
      continue;
    }

    if(initDone && Options::interactive) 
    {
      if(FD_ISSET(0,&rfds) ) 
      {   //stdin
        cin.getline(buffer,bufferMaxSize);
        if(strlen(buffer)>0) 
        {
          int mod;
          for(mod=0;mod<MOD::numModules;mod++) 
          {
            bool ret=MOD::coachModule[mod]->onKeyboardInput(buffer);
            if(ret) break;
          }
          if(mod==MOD::numModules) 
          {
            cout << "\nCommand not understood." << flush;
          }
        }
        //sock.send_msg(buffer,strlen(buffer));
      }
    }

    if(FD_ISSET(sock.get_socket_fd(),&rfds))
    {
      while ((msgType=MSG::recvAndParseMsg())>0) 
      {
        if (!initDone) 
        {
          initPostConnect(argc,argv);
          continue;
        }

        if (Options::synch_mode) 
        {

          if(msgType==MSG::MSG_HEAR) 
          {
            for(int mod=0;mod<MOD::numModules;mod++) 
            {
              MOD::coachModule[mod]->onHearMessage(MSG::msgBuffer);
            }
          }

          //if (msgType != MSG::MSG_THINK) continue; //ACHTUNG: dadurch werden hear messages nicht mehr geparst!!!
          if (    ((MSG::msg[MSG::MSG_THINK]->lastRcvd != fld.getTime())
               && (fld.getTime() != 0))
               || ((MSG::msg[MSG::MSG_DONE]->lastSent == fld.getTime())
               && (fld.getTime() != 0)) )
          {
            continue;
          }

          if ( (last_loop_time == fld.getTime()) && (fld.getTime() != 0)) 
          {
            continue;
          }

          /** main behave loop */
          last_loop_time = fld.getTime();

          for (int mod=0;mod<MOD::numModules;mod++) 
          {
            MOD::coachModule[mod]->behave();
          }

          if(fld.getPM()!=oldPM) 
          { 
            for(int mod=0;mod<MOD::numModules;mod++) 
            {
              MOD::coachModule[mod]->onRefereeMessage(fld.getPM()<PM_MAX);
            }
          }
        } //end of Options::synch_mode
        else 
        {
          if (msgType==MSG::MSG_SEE_GLOBAL) 
          {	  
            /** main behave loop */
            for(int mod=0;mod<MOD::numModules;mod++) 
            {
              MOD::coachModule[mod]->behave();
            }
          }
          if (fld.getPM()!=oldPM) 
          {
            for(int mod=0;mod<MOD::numModules;mod++) 
            {
              MOD::coachModule[mod]->onRefereeMessage(fld.getPM()<PM_MAX);
            }
          } 
          else 
          {
            if (msgType==MSG::MSG_HEAR) 
            {
              for(int mod=0;mod<MOD::numModules;mod++) 
              {
                MOD::coachModule[mod]->onHearMessage(MSG::msgBuffer);
              }
            }
          }
        } //end of else (i.e. !Options::synch_mode)
        oldPM=fld.getPM();	
      }
    }
    if (    (Options::synch_mode) 
         && (MSG::msg[MSG::MSG_THINK]->lastRcvd == fld.getTime()) 
         && (MSG::msg[MSG::MSG_DONE]->lastSent != fld.getTime()) ) 
    {
      MSG::sendMsg(MSG::MSG_DONE);
    }
  } //end of while(serverAlive)
}
