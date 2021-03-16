#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include "globaldef.h"
#include "Vector.h"
#include <fstream>
#include "server_options.h" //should be removed from here and other files  should be forced to include server_options.h directly <art>

struct CommandLineOptions {
	static const int MAX_LEN= 150;
	static char host[MAX_LEN];
	static int  port;
	static char agent_conf[MAX_LEN];
	static char log_dir[MAX_LEN];
	static char server_conf[MAX_LEN];
	static char formations_conf[MAX_LEN];
	static char strategy_conf[MAX_LEN];
	static char policy_conf[MAX_LEN];
	static char moves_conf[MAX_LEN];
	static char behavior_conf[MAX_LEN];
	/** the server_version (default 5.24)*/
	static int cheat_level;
	static int asynchron;  //@andi: flag fuer den asynchronen Server
	static int neck_lock;  /** enable neck locking for Face_Ball */
	static void init_options();
	static bool read_from_command_line(int argc, char **argv);
	static void print_options();
private:
	static const int MAX_LINE_LENGTH = 500;
};

class ClientOptions {
public:
	static void init();
	static bool read_from_file(char *config_file);
	static bool read_from_command_line(int argc, char **argv);

	// Configurable options
	static char teamname[20];
	static char body_behavior[60];
	static int strategy_type;
	static int neck_type; /** type of turn_neck strategy */
	static int use_neckview;
	static int view_type;
	/** flag to check whether we are a goalie */
	static int consider_goalie;

	static double server_version;
	static int neck_1v1;

	// Not configurable options
	static int player_no;
};

/*****************************************************************************/
/*****************************************************************************/
/* LogOptions */

class LogOptions{
	struct OutOpt {
		void init(bool a, bool b, bool c) {
			log_cout= a;
			log_cerr= b;
			log_file= c;
		}
		void init(int num, const char * val);
		OutOpt() { init(false,false,false); }
		OutOpt(bool a, bool b, bool c) { init(a,b,c); }
		bool log_cout;
		bool log_cerr;
		bool log_file;
	};
	static bool use_file() { return opt_log[xx_DEF].log_file || opt_log[xx_ERR].log_file || opt_log[xx_DEB].log_file || opt_log[xx_WM].log_file || opt_log[xx_MDP].log_file || log_int; }
	static const int MAX_LEN= 512;
public:
	static char log_dir[MAX_LEN];
	static char log_fname[MAX_LEN];
	static void init();
	static void open();
	static void close();
	static bool read_from_command_line(int argc, char **argv);
	static bool read_from_file( char * config_file );

	static const int num_level= 6;
	static const int xx_DEF= 0;
	static const int xx_ERR= 1;
	static const int xx_DEB= 2;
	static const int xx_WM=  3;
	static const int xx_MDP= 4;
	static const int xx_NUM= 5;
	static const char indent[xx_NUM][num_level][num_level+1];
	static int max_level;
	static int log_stat;
	static int log_gStat; // goalsStatistic.
	static OutOpt opt_log[xx_NUM];
	static bool log_int;

	static int vis_offside; /** visualize offside lines */

	static std::ofstream file;
	static bool is_off() { return max_level < 0; }
	static bool is_on() { return !is_off(); }
};

#endif

