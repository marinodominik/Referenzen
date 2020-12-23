#include "options.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include "valueparser.h"

/*****************************************************************************/
char CommandLineOptions::host[MAX_LEN];
int  CommandLineOptions::port=6000;
char CommandLineOptions::agent_conf[MAX_LEN];   
char CommandLineOptions::server_conf[MAX_LEN];  
char CommandLineOptions::formations_conf[MAX_LEN];
char CommandLineOptions::strategy_conf[MAX_LEN];
char CommandLineOptions::policy_conf[MAX_LEN];  
char CommandLineOptions::moves_conf[MAX_LEN];
char CommandLineOptions::behavior_conf[MAX_LEN];
char CommandLineOptions::log_dir[MAX_LEN];   
int CommandLineOptions::cheat_level;
int CommandLineOptions::asynchron;
int CommandLineOptions::neck_lock;

void CommandLineOptions::init_options() {
	port= 6000;
}

bool CommandLineOptions::read_from_command_line(int argc,char **argv) {
	/* skip command name */
	argv++ ; argc-- ;

	bool dum;
	ValueParser vp(argc,argv);
	//vp.set_verbose();
	if ( vp.get("help",dum) >= 0 ) {
		cout << "\n--------------------------------------------------------------------------\n\
1st usage:\n\
 FRA-UNIted_Agent\n\
   -help          this page\n\
--\n\
2nd usage:\n\
 FRA-UNIted_Agent\n\
   -host            <host you wish to connect to>\n\
   -port            <port at host>\n\
   -agent_conf      <name of file containing agent configuration>\n\
   -server_conf     <name of file containing server configuration>\n\
   -formations_conf <name of file containing configuration for formations>\n\
   -log_dir         <directory to save log files>   \n\
   -log_lev         -1= no logging, 0= level 0, 1= level 0 and 1\n\
   -log_stat        1= log MDP statistics to screen, 0= logfile only\n\
   -log_gStat       goals Statistic 0= do nothing, 1=via log_mdp, 2=to screen and via log_mdp\n\
\n\
   s= standard output (cout), e= error output (cerr), f= file output\n\
   1= turn on, 0= turn off, *= let default value\n\
   log_def means LOG_STR, LOG_POL, LOG_MOV, LOG_MDP\n\
   -log_def         sef   s,e,f in  {0,1,*}\n\
   -log_err         sef   s,e,f in  {0,1,*}\n\
   -log_deb         sef   s,e,f in  {0,1,*}\n\
   -log_wm          sef   s,e,f in  {0,1,*}\n\
   -log_mdp         sef   s,e,f in  {0,1,*}\n\
\n\
   -neck_lock       enable neck locking in Move_Face_Ball (default=0)\n\
\n\
   -cheat           possible values are 0= regular game, \n\
                                        1= record mdp info, 2= use mpd info\n\
   -async           possible values are 0=synchronous server\n\
                                        1=asynchronous server\n\
\n\
   options which override the values of options from agent_conf\n\
   -t_name        <team name>\n\
   -s_type        <strategy type>\n\
   -view_type     <view strategy>\n\
   -neck_type     <turn_neck strategy>\n\
   -goalie        flag for using the goalie \n\
--\n\
  \n\
 Order of options doesn't matter!\n\
 A missing option does take a default value, see below list:\n\
   -host            localhost\n\
   -agent_conf      ./conf/agent.conf\n\
   -server_conf     ./conf/server.conf\n\
   -formations_conf ./conf/formations_conf\n\
\n\
   -log_dir         .\n\
   -log_lev         -1\n\
   -log_stat        0\n\
   -log_gStat       0\n\
   -log_def         000\n\
   -log_err         010\n\
   -log_deb         100\n\
\n\
   -cheat           0\n\
   -async           0\n\
   \n\
--------------------------------------------------------------------------"
				<< std::endl;
		exit(0);
	}
	vp.get("host",host,MAX_LEN,"localhost");
	vp.get("port",port);
	vp.get("agent_conf",agent_conf,MAX_LEN,"../conf/agent.conf");
	vp.get("server_conf",server_conf,MAX_LEN,"../conf/server.conf");
	vp.get("formations_conf",formations_conf,MAX_LEN,"../conf/formations.conf");
	//vp.get("strategy_conf",strategy_conf,MAX_LEN,"../conf/strategy.conf");
	strncpy(strategy_conf,agent_conf,MAX_LEN);
	//vp.get("policy_conf",policy_conf,MAX_LEN,"../conf/policy.conf");
	strncpy(policy_conf,agent_conf,MAX_LEN);
	//vp.get("moves_conf",moves_conf,MAX_LEN,"../conf/moves.conf");
	strncpy(moves_conf,agent_conf,MAX_LEN);
	//vp.get("behavior_conf",moves_conf,MAX_LEN,"../conf/behavior.conf");
	strncpy(behavior_conf,agent_conf,MAX_LEN);
	vp.get("log_dir",log_dir,MAX_LEN,".");
	vp.get("async", asynchron, 0);
	vp.get("neck_lock",neck_lock,0);
	vp.get("cheat",cheat_level,0);
	return true;
}

void CommandLineOptions::print_options()  {
	std::cout //<< "\n   " // CommandLineOptions:"
			//       << "\n   host            = " << host
			<< "\n   agent_conf      = " << agent_conf
			//       << "\n   server_conf     = " << server_conf
			<< "\n   formations_conf = " << formations_conf
			//       << "\n   strategy_conf   = " << strategy_conf
			//       << "\n   policy_conf     = " << policy_conf
			//       << "\n   moves_conf      = " << moves_conf
			<< "\n   log_dir         = " << log_dir
			//       << "\n   cheat           = " << cheat_level
			//       << "\n   neck_lock       = " << neck_lock
			<< std::endl;
}

/*****************************************************************************/
char   ClientOptions::teamname[20];
char   ClientOptions::body_behavior[60];
int    ClientOptions::strategy_type;
int    ClientOptions::neck_type;
int    ClientOptions::use_neckview;
int    ClientOptions::view_type;
int    ClientOptions::consider_goalie;
int    ClientOptions::player_no;
double ClientOptions::server_version;
int    ClientOptions::neck_1v1;


void ClientOptions::init() {
	strategy_type= 0;
	neck_type= 40;
	neck_type= 1;
	view_type= SYNCH_SEE;
	strncpy(teamname,"FRA-UNIted",20);
	body_behavior[0]='\0';
	consider_goalie= 0;
	server_version = 12.0;
	neck_1v1=0;
}

bool ClientOptions::read_from_file(char *config_file){
	ValueParser vp(config_file);
	//  vp.set_verbose();
	vp.get("strategy_type",strategy_type);
	vp.get("neck_type",neck_type);
	vp.get("use_neckview",use_neckview);
	vp.get("view_type",view_type);
	vp.get("teamname",teamname,20);
	vp.get("body_behavior",body_behavior,60);
	vp.get("consider_goalie",consider_goalie);
	vp.get("server_version",server_version);
	vp.get("neck_1v1",neck_1v1);
	//cout << "\n   neck_type (file) = " << neck_type << std::endl;
	return true;
}

bool ClientOptions::read_from_command_line(int argc, char **argv){
	/* skip prgram name */
	argv++ ; argc-- ;

	int dum;
	ValueParser vp(argc,argv);
	vp.get("s_type",strategy_type);
	vp.get("t_name",teamname,20);
	vp.get("body_behavior",body_behavior,60);
	vp.get("neck_type",neck_type);
	vp.get("view_type",view_type);
	if ( vp.get("goalie",dum)>= 0 )
		consider_goalie= 1;
	vp.get("server_version",server_version);
	vp.get("neck_1v1",neck_1v1);
#if 0
	std::cout << "\n   " // CommandLineOptions:"
			<< "\n   view_type       = " << view_type
			<< "\n   neck_type       = " << neck_type
			<< std::endl;
#endif
	return true;
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/* LogOptions */

void LogOptions::OutOpt::init(int num, const char * val) {
	if (num>0)
	{
		if ( val[0] == '1' )
			log_cout= true;
		else if ( val[0] == '0' )
			log_cout= false;
	}

	if (num>1)
	{
		if ( val[1] == '1' )
			log_cerr= true;
		else if ( val[1] == '0' )
			log_cerr= false;
	}

	if (num>2)
	{
		if ( val[2] == '1' )
			log_file= true;
		else if ( val[2] == '0' )
			log_file= false;
	}
}

const char LogOptions::indent[xx_NUM][num_level][num_level+1]= {
		{ "-","--","---","----","-----","------" },
		{ "#","#","#","#","#","#" },
		//  { "-","-", "-", "-", "-", "-"},
		{ "-","--","---","----","-----","------" },
		{ "-","--","---","----","-----","------" },
		{ "-","--","---","----","-----","------" }
};

char LogOptions::log_dir[MAX_LEN];
char LogOptions::log_fname[MAX_LEN];

int LogOptions::max_level;
LogOptions::OutOpt LogOptions::opt_log[xx_NUM];

int LogOptions::log_stat;
int LogOptions::log_gStat;
int LogOptions::vis_offside;
bool LogOptions::log_int;
std::ofstream LogOptions::file;

void LogOptions::init() {
	max_level= -1;
	log_stat=0;
	log_gStat=0;
	opt_log[xx_DEF].init(false,false,false);
	opt_log[xx_ERR].init(false,true,false);
	opt_log[xx_DEB].init(true,false,false);
	opt_log[xx_WM].init(false,false,false);
	opt_log[xx_MDP].init(false,false,false);

	log_int=false; // ridi 25.5.02
	vis_offside=0;
}

void LogOptions::close() {
	if (use_file())
		file.close();
}

bool LogOptions::read_from_command_line(int argc, char **argv) {
	/* skip command name */
	argv++ ; argc-- ;

	char dum[3], logDirAtCmdLine[MAX_LEN];
	int num, read_level;
	ValueParser vp(argc,argv);
	//vp.set_verbose();
//	std::cout<<"LogOptions from command line: ";
	num= vp.get("log_def",dum,3);
	if (num==3) {
		opt_log[xx_DEF].init(num,dum);
//		std::cout<<"log_def "<<dum<<" ";
	}
	num= vp.get("log_err",dum,3);
	if (num==3) {
		opt_log[xx_ERR].init(num,dum);
//		std::cout<<"log_err "<<dum<<" ";
	}
	num= vp.get("log_deb",dum,3);
	if (num==3) {
		opt_log[xx_DEB].init(num,dum);
//		std::cout<<"log_deb "<<dum<<" ";
	}
	num= vp.get("log_wm",dum,3);
	if (num==3) {
		opt_log[xx_WM].init(num,dum);
//		std::cout<<"log_wm "<<dum<<" ";
	}
	num= vp.get("log_mdp",dum,3);
	if (num==3) {
		opt_log[xx_MDP].init(num,dum);
//		std::cout<<"log_mdp "<<dum<<" ";
	}

	num= vp.get("log_lev",read_level);
	if (num>0) {
		max_level = read_level;
//		std::cout<<"log_lev "<<max_level<<" ";
	}
	if (log_stat == 0)  num= vp.get("log_stat",log_stat);
	if (log_gStat == 0) num= vp.get("log_gStat",log_gStat);
	num= vp.get("log_dir",logDirAtCmdLine,MAX_LEN);
	if (num>0) {
		strncpy(log_dir, logDirAtCmdLine, MAX_LEN);
//		std::cout<<"log_dir "<<log_dir<<" "<<endl;
	}

	if( vp.get("log_int",log_int)== 0 )
		log_int=true;

	if( vp.get("vis_offside",num)== 0 )
		vis_offside=1;

	return true;
}

void LogOptions::open() { 
	if ( use_file() ) {
		file.open(log_fname);
		//cout << "\nopening file " << log_fname;
	}
}

/*
int main() {
  ServerOptions::init();
  printf("\n\n %d \n\n",(ServerOptions::verbose));
  return 0;
}
 */
bool LogOptions::read_from_file( char * config_file )
{
	ValueParser vp(config_file);
	char dum[3];
	int num;

//	std::cout<<"LogOptions from config file: ";
	num= vp.get("log_def",dum,3); opt_log[xx_DEF].init(num,dum);
//	if (num>0) std::cout<<"log_def "<<dum<<" ";
	num= vp.get("log_err",dum,3); opt_log[xx_ERR].init(num,dum);
//	if (num>0) std::cout<<"log_err "<<dum<<" ";
	num= vp.get("log_deb",dum,3); opt_log[xx_DEB].init(num,dum);
//	if (num>0) std::cout<<"log_deb "<<dum<<" ";
	num= vp.get("log_wm",dum,3);  opt_log[xx_WM].init(num,dum);
//	if (num>0) std::cout<<"log_wm "<<dum<<" ";
	num= vp.get("log_mdp",dum,3);  opt_log[xx_MDP].init(num,dum);
//	if (num>0) std::cout<<"log_mdp "<<dum<<" ";

	num= vp.get("log_lev",max_level);
//	if (num>0) std::cout<<"log_lev "<<max_level<<" ";
	num= vp.get("log_stat",log_stat);
	num= vp.get("log_gStat",log_gStat);
	num= vp.get("log_dir",log_dir,MAX_LEN,"./");
//	if (num>0) std::cout<<"log_dir "<<log_dir<<" ";
//	std::cout<<endl;

	if( vp.get("log_int",log_int)== 0 )
		log_int=true;

	if( vp.get("vis_offside",num)== 0 )
		vis_offside=1;

	return true;

}
