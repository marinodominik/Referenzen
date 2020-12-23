#include "options.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include "valueparser.h"
#include <stdlib.h>


/*****************************************************************************/
char Options::host[MAX_LEN];
char Options::coach_conf[MAX_LEN];
int Options::port;
bool Options::isTrainer;
bool Options::waitForOurTeam;
bool Options::waitForHisTeam;
bool Options::synch_mode;
bool Options::interactive;
char Options::teamName[30];

bool Options::server_94;
int  Options::cvServerProtocol=14;

void Options::init() {
  isTrainer=false;
  synch_mode = false;
}

bool Options::read_config_name(int argc,char **argv) {
  argv++ ; argc-- ;
  ValueParser vp(argc,argv);
  vp.get("conf",coach_conf,MAX_LEN,"../conf/coach.conf");
  return true;
}

bool Options::read_from_command_line(int argc,char **argv) {
  /* skip command name */
  argv++ ; argc-- ;

  bool dum;
  ValueParser vp(argc,argv);
  //vp.set_verbose();
  if ( vp.get("help",dum) >= 0 ) {
    cout << "\n-------------------------------------------------------------------------"
      "\n                              FRA-UNIted_Coach"
      "\n                              ================"
      "\n Possible command line options are:"
      "\n   -host <servername>   [localhost]"
      "\n   -port <serverport>   [6002]"
      "\n   -conf <config file>  [coach.conf]"
      "\n   -isTrainer           [0]"
      "\n   -teamName <teamname> [BS01]"
      "\n   -interactive         [0]"
      "\n   -waitForOurTeam      [1]"
      "\n   -waitForHisTeam      [1]"
      "\n   -synch_mode          [0]"
      "\n"
   
      "\n--------------------------------------------------------------------------\n"
	   << endl;
      exit(0);
  }

  if(vp.get("trainer",isTrainer)>=0) isTrainer=true;
  vp.get("host",host,MAX_LEN);
  vp.get("port",port,42);
  //vp.get("conf",coach_conf,MAX_LEN,"../conf/coach.conf");
  vp.get("teamName",teamName,30);
  vp.get("interactive",interactive);
  vp.get("waitForOurTeam",waitForOurTeam);
  vp.get("waitForHisTeam",waitForHisTeam);
  vp.get("synch_mode", synch_mode);
  if(vp.get("server_9.4", server_94)>=0) server_94=true;else server_94=false;
  
  if(port==42) {port=isTrainer ? 6001 : 6002;}
  return true;
}

bool Options::read_from_file(char *config_file){
  ValueParser vp(config_file,"Global");
  //  vp.set_verbose();
  vp.get("teamName",teamName,30,"BS03");
  vp.get("host",host,MAX_LEN,"localhost");
  vp.get("port",port,42);
  int dum=0;
  isTrainer=false;
  if(vp.get("isTrainer",dum)>=0) {
    isTrainer=dum?true:false;
  }
  vp.get("interactive",interactive,0);
  vp.get("waitForOurTeam",waitForOurTeam,1);
  vp.get("waitForHisTeam",waitForHisTeam,1);
  vp.get("synch_mode", synch_mode, 0);
  if(vp.get("server_9.4", dum)>=0)
    server_94 = dum ? true : false;

  if(port==42) {port=isTrainer ? 6001 : 6002;}
  return true;
}


void Options::print_options()  {
  cout << "\n  Options:\n"
       << "\n    host           = " << host
       << "\n    port           = " << port
       << "\n    conf           = " << coach_conf
       << "\n    teamName       = " << teamName
       << "\n    trainer        = " << isTrainer
       << "\n    interactive    = " << interactive
       << "\n    waitForOurTeam = " << waitForOurTeam
       << "\n    waitForHisTeam = " << waitForHisTeam
       << "\n    synch_mode     = " << synch_mode
       << endl;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/* LogOptions */

void LogOptions::OutOpt::init(int num, const char * val) {
  if (num>0){
    if ( val[0] == '1' )
      log_cout= true;
    else if ( val[0] == '0' )
      log_cout= false;
  }
  if (num>1) {
    if ( val[1] == '1' )
      log_cerr= true;
    else if ( val[1] == '0' )
      log_cerr= false;
  }
  if (num>2) {
    if ( val[2] == '1' )
      log_file= true;
    else if ( val[2] == '0' )
      log_file= false;
  }
}

const char LogOptions::indent[xx_NUM][num_level][num_level+6]= { 
  { "-","--","---" },
  { "#","#","#" },
  { "- ","- ", "- "},
  { "- MSG:","-- MSG:","--- MSG:" },
  { "- FLD:","-- FLD:","--- FLD:" }, 
  { "- INT:","-- INT:","--- INT:" } 
};

char LogOptions::log_dir[MAX_LEN];
char LogOptions::log_fname[MAX_LEN];

int LogOptions::max_level;
LogOptions::OutOpt LogOptions::opt_log[xx_NUM];

ofstream LogOptions::file;

void LogOptions::init() {
  max_level= 2;
  sprintf(log_dir,"./log");
  opt_log[xx_DEF].init(true,false,false);
  opt_log[xx_ERR].init(true,false,false);
  opt_log[xx_VIS].init(false,false,true);
  opt_log[xx_MSG].init(true,false,true);
  opt_log[xx_FLD].init(true,false,true);
  opt_log[xx_INT].init(true,false,true);

}

void LogOptions::close() {
  if (use_file())
    file.close();
}

bool LogOptions::read_from_command_line(int argc, char **argv) {
  /* skip command name */
  argv++ ; argc-- ;

  ValueParser vp(argc,argv);
  readOptions(&vp);

  return true;
}

bool LogOptions::read_from_file(char *config_file){
  ValueParser vp(config_file,"LogOptions");
  readOptions(&vp);
  return true;
}

void LogOptions::readOptions(ValueParser *vp) {
  char dum[3];
  int num;

  //vp.set_verbose();
  num= vp->get("log_def",dum,3);  opt_log[xx_DEF].init(num,dum);
  num= vp->get("log_err",dum,3);  opt_log[xx_ERR].init(num,dum);
  num= vp->get("log_vis",dum,3);  opt_log[xx_VIS].init(num,dum);
  num= vp->get("log_msg",dum,3);  opt_log[xx_MSG].init(num,dum);
  num= vp->get("log_fld",dum,3);  opt_log[xx_FLD].init(num,dum);
  num= vp->get("log_int",dum,3);  opt_log[xx_INT].init(num,dum);

  num= vp->get("log_lev",max_level,2);
  num= vp->get("log_dir",log_dir,MAX_LEN);  
}
  
void LogOptions::open() { 
  if ( use_file() ) {
    file.open(log_fname);
//    cout << "\nopening file " << log_fname;
  }
}
