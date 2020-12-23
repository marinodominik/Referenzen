#include "server_options.h"

double ServerOptions::ball_accel_max          =      2.7;
double ServerOptions::ball_decay              =      0.94;
double ServerOptions::ball_rand               =      0.05;
double ServerOptions::ball_size               =      0.085;
double ServerOptions::ball_speed_max          =      3;
double ServerOptions::catchable_area_l        =      1.2;
double ServerOptions::catchable_area_w        =      1;
int    ServerOptions::catch_ban_cycle         =      5;
double ServerOptions::dash_power_rate         =      0.006;
int    ServerOptions::drop_ball_time          =    100;
double ServerOptions::effort_dec              =      0.005;
double ServerOptions::effort_dec_thr          =      0.3;
double ServerOptions::effort_inc              =      0.01;
double ServerOptions::effort_inc_thr          =      0.6;
double ServerOptions::effort_min              =      0.6;
bool   ServerOptions::fullstate_l             =      0;
bool   ServerOptions::fullstate_r             =      0;
double ServerOptions::goal_width              =     14.02;
int    ServerOptions::half_time               =    300;
double ServerOptions::inertia_moment          =      5;
double ServerOptions::kickable_margin         =      0.7;
double ServerOptions::kick_power_rate         =      0.027;
ANGLE  ServerOptions::maxneckang;
ANGLE  ServerOptions::maxneckmoment;
double ServerOptions::maxpower                =    100;
ANGLE  ServerOptions::minneckang;
ANGLE  ServerOptions::minneckmoment;
double ServerOptions::minpower                =   -100;
double ServerOptions::player_decay            =      0.4;
double ServerOptions::player_rand             =      0.1;
double ServerOptions::player_size             =      0.3;
double ServerOptions::player_speed_max        =      1.05;
double ServerOptions::recover_dec             =      0.002;
double ServerOptions::recover_dec_thr         =      0.3;
double ServerOptions::recover_min             =      0.5;
int    ServerOptions::simulator_step          =    100;
int    ServerOptions::send_step               =    150;
int    ServerOptions::slow_down_factor        =      1;
double ServerOptions::stamina_max             =   8000;
double ServerOptions::stamina_inc_max         =     45;
bool   ServerOptions::use_offside             =      1;
double ServerOptions::visible_angle           =     90;
double ServerOptions::visible_distance        =      3;
double ServerOptions::tackle_dist             =      2;
double ServerOptions::tackle_back_dist        =      0;
double ServerOptions::tackle_width            =      1.25;
int    ServerOptions::tackle_exponent         =      6;
int    ServerOptions::tackle_cycles           =     10;
double ServerOptions::player_speed_max_min    =      0.75;
double ServerOptions::max_tackle_power        =    100;
double ServerOptions::max_back_tackle_power   =      0.0;
int    ServerOptions::extra_stamina           =     50.0;
int    ServerOptions::ball_stuck_area         =      3;
int    ServerOptions::synch_see_offset        =      0;
bool   ServerOptions::pen_allow_mult_kicks    =      1;
int    ServerOptions::pen_nr_kicks            =      5;
int    ServerOptions::pen_max_extra_kicks     =     10;
double ServerOptions::pen_max_goalie_dist_x   =     14;
double ServerOptions::pen_dist_x              =     42.5;
double ServerOptions::tackle_power_rate       =      0.027;
// JTS: new ones for V13
double ServerOptions::stamina_capacity        = 130600.0; // JTS 10 --> changed param
double ServerOptions::max_dash_angle          =    180.0;
double ServerOptions::min_dash_angle          =   -180.0;
double ServerOptions::dash_angle_step         =     45.0; // JTS 10 changed side_dash related to new defaults
double ServerOptions::side_dash_rate          =      0.4;
double ServerOptions::back_dash_rate          =      0.6;
double ServerOptions::max_dash_power          =    100.0;
double ServerOptions::min_dash_power          =   -100.0;
int    ServerOptions::extra_half_time         =    100;
// JTS 10: new ones for V14
double ServerOptions::tackle_rand_factor      =      2.0;
double ServerOptions::foul_detect_probability =      0.5;
double ServerOptions::foul_exponent           =     10.0;
int    ServerOptions::foul_cycles             =      5;
int    ServerOptions::golden_goal             =      1;

//not used ones:
//double ServerOptions::ball_weight= 0.2;
//double ServerOptions::catch_probability= 1.0;
//int ServerOptions::goalie_max_moves= 2;
//ServerOptions::kickable_area is defined at the end;
//double ServerOptions::player_accel_max;
//double ServerOptions::player_weight;

double ServerOptions::kickable_area = ServerOptions::kickable_margin + ServerOptions::ball_size + ServerOptions::player_size;

/* these are not from server.conf... */
double ServerOptions::penalty_area_width      =    40.32;
double ServerOptions::penalty_area_length     =    16.5;
double ServerOptions::pitch_length            =   105.0;
double ServerOptions::pitch_width             =    68.0;
Vector ServerOptions::own_goal_pos            = Vector( -52.5,  0.0  );
Vector ServerOptions::their_goal_pos          = Vector(  52.5,  0.0  );
Vector ServerOptions::their_left_goal_corner  = Vector(  52.5,  7.01 );
Vector ServerOptions::their_right_goal_corner = Vector(  52.5, -7.01 );

#if 0
/* read from is no longer supported because this method is way to old
   and we are parsing everything directly out of the server messages
   since v8 of soccerserver 
   WARNING: IF YOU WANT TO USE THIS PLEASE MAKE SURE TO WRITE A NEW FUNCTION
   THAT IS ABLE TO INITIALIZE EVERY PARAMETER LISTED ABOVE
 */
bool ServerOptions::read_from_file(char *config_file){
  ERROR_OUT << "read from file no longer supported";
  exit(1);
  FILE *server_conf;
  server_conf = fopen(config_file,"r");
  if (server_conf == NULL) {
    cerr << "\nCan't open file " << config_file 
	 << "\n>>>  This file should contain parameters of the current server!!!" << endl;
    return false;
  }
  //cout << "\nReading server's parameter from " << config_file << endl;
  char line[1000];
  char com[256];
  int n;
  while(fgets(line,1000,server_conf)!=NULL){
    n = sscanf(line,"%s", com) ;
    char *t = line ;
#define NULLCHAR        '\000'
    while(*t != NULLCHAR) {
      if (*t == ':') *t = ' ' ;
      t++ ;
    }
    double lf;
    if(*line=='\n'||*line=='#');  /* skip comments */
    else if (strncasecmp(line,"goal_width",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      goal_width= double(lf);
    } 
    else if (strncasecmp(line,"penalty_area_width",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      penalty_area_width= double(lf);
    } 
    else if (strncasecmp(line,"penalty_area_length",10)==0){
      n = sscanf(line, "%s %lf", com, &lf) ;
      penalty_area_length = double(lf);
    } 
    else if (strncasecmp(line,"player_size",11)==0){
      n = sscanf(line, "%s %lf", com, &lf) ;
      player_size= double(lf);
    } 
    else if (strncasecmp(line,"player_decay",12)==0){
      n = sscanf(line, "%s %lf", com, &lf) ;
      player_decay=  double(lf);
    } 
    else if (strncasecmp(line,"player_rand",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      player_rand= double(lf);
    } 
    else if (strncasecmp(line,"player_weight",13)==0){
      n = sscanf(line, "%s %lf", com, &lf) ;
      //player_weight = double(lf);
    } 
    else if (strncasecmp(line,"player_speed_max",16)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      player_speed_max= double(lf);
    }
    else if (strncasecmp(line,"player_accel_max",16)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //player_accel_max= double(lf);
    }
    else if (strncasecmp(line,"stamina_max",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      stamina_max= double(lf);
    } 
    else if (strncasecmp(line,"stamina_inc_max",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      stamina_inc_max= double(lf);
    } 
    else if (strncasecmp(line,"recover_dec_thr",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      recover_dec_thr= double(lf);
    } 
    else if (strncasecmp(line,"recover_dec",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      recover_dec= double(lf);
    } 
    else if (strncasecmp(line,"recover_min",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      recover_min= double(lf);
    } 
    else if (strncasecmp(line,"effort_dec_thr",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      effort_dec_thr= double(lf);
    } 
    else if (strncasecmp(line,"effort_dec",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      effort_dec= double(lf);
    } 
    else if (strncasecmp(line,"effort_inc_thr",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      effort_inc_thr= double(lf);
    } 
    else if (strncasecmp(line,"effort_inc",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      effort_inc= double(lf);
    } 
    else if (strncasecmp(line,"effort_min",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      effort_min= double(lf);
    } 
    else if (strncasecmp(line,"hear_max",8)==0){
      //n = sscanf(line, "%s %d", com, &hear_max ) ;
    } 
    else if (strncasecmp(line,"hear_inc",8)==0){
      //n = sscanf(line, "%s %d", com, &hear_inc ) ;
    } 
    else if (strncasecmp(line,"hear_decay",10)==0){
      //n = sscanf(line, "%s %d", com, &hear_decay ) ;
    } 
    else if (strncasecmp(line,"inertia_moment",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      inertia_moment= double(lf);
    } 
    else if (strncasecmp(line,"catchable_area_l",16)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      catchable_area_l= double(lf);
    } 
    else if (strncasecmp(line,"catchable_area_w",16)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //catchable_area_w= double(lf);
    } 
    else if (strncasecmp(line,"catch_probability",17)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //catch_probability= double(lf);
    } 
    else if (strncasecmp(line,"catch_ban_cycle",15)==0){
      n = sscanf(line, "%s %d", com, &catch_ban_cycle ) ;
    }
    else if (strncasecmp(line,"goalie_max_moves",16)==0){
      //n = sscanf(line, "%s %d", com, &goalie_max_moves ) ;
    }
    else if (strncasecmp(line,"ball_size",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      ball_size= double(lf);
    } 
    else if (strncasecmp(line,"ball_decay",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      ball_decay= double(lf);
    } 
    else if (strncasecmp(line,"ball_rand",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      ball_rand= double(lf);
    } 
    else if (strncasecmp(line,"ball_weight",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //ball_weight= double(lf);
    } 
    else if (strncasecmp(line,"ball_speed_max",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      ball_speed_max= double(lf);
    }
    else if (strncasecmp(line,"ball_accel_max",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      ball_accel_max= double(lf);
    }
    else if (strncasecmp(line,"wind_force",10)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //wind_force= double(lf);
    } 
    else if (strncasecmp(line,"wind_dir",8)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //wind_dir= double(lf);
    } 
    else if (strncasecmp(line,"wind_rand",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //wind_rand= double(lf);
    } 
    else if (strncasecmp(line,"kickable_margin",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      kickable_margin= double(lf);
    } 
    else if (strncasecmp(line,"ckick_margin",12)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //ckick_margin= double(lf);
    }
    else if (strncasecmp(line,"kick_rand",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //kick_rand= double(lf);
    }
    else if (strncasecmp(line,"dash_power_rate",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      dash_power_rate= double(lf);
    } 
    else if (strncasecmp(line,"kick_power_rate",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      kick_power_rate= double(lf);
    } 
    else if (strncasecmp(line,"visible_angle",13)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      visible_angle= double(lf);
    } 
    else if (strncasecmp(line,"audio_cut_dist",14)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //audio_cut_dist= double(lf);
    } 
    else if (strncasecmp(line,"quantize_step",13)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //quantize_step= double(lf);
    } 
    else if (strncasecmp(line,"quantize_step_l",15)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //quantize_step_l= double(lf);
    } 
    else if (strncasecmp(line,"maxpower",8)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      maxpower= double(lf);
    } 
    else if (strncasecmp(line,"minpower",8)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      minpower= double(lf);
    } 
    else if (strncasecmp(line,"maxmoment",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //maxmoment= double(lf);
    } 
    else if (strncasecmp(line,"minmoment",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //minmoment= double(lf);
    } 
    else if (strncasecmp(line,"port",4)==0){
      //n = sscanf(line, "%s %d", com, &port ) ;
    } 
    else if (strncasecmp(line,"coach_port",10)==0){
      //n = sscanf(line, "%s %d", com, &coach_port ) ;
    } 
    else if (strncasecmp(line,"simulator_step",14)==0){
      n = sscanf(line, "%s %d", com, &simulator_step ) ;
    } 
    else if (strncasecmp(line,"send_step",9)==0){
      //n = sscanf(line, "%s %d", com, &send_step ) ;
    } 
    else if (strncasecmp(line,"recv_step",9)==0){
      //n = sscanf(line, "%s %d", com, &recv_step ) ;
    } 
    else if (strncasecmp(line,"half_time",9)==0){
      n = sscanf(line, "%s %d", com, &half_time ) ;
    } 
    else if (strncasecmp(line,"say_msg_size",12)==0){
      //n = sscanf(line, "%s %d", com, &say_msg_size ) ;
    } 
    else if (strncasecmp(line,"use_offside",11)==0){
      char use_offside_string[128];
      n = sscanf(line, "%s %s", com, use_offside_string ) ;
      use_offside = (!strncasecmp(use_offside_string, "on",2)) ? true : false ;
    } 
    else if (strncasecmp(line,"offside_active_area_size",24)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //offside_active_area_size= double(lf);
    } 
    else if (strncasecmp(line,"forbid_kick_off_offside",23)==0){
      char forbid_kick_off_offside_string[128];
      n = sscanf(line, "%s %s", com, forbid_kick_off_offside_string ) ;
      //forbid_kick_off_offside = (!strncasecmp(forbid_kick_off_offside_string, "on",2)) ? true : false ;
      // cout << forbid_kick_off_offside << endl;
    } 
    else if (strncasecmp(line,"verbose",7)==0){
      char verbose_string[128];
      n = sscanf(line, "%s %s", com, verbose_string ) ;
      //verbose = (!strncasecmp(verbose_string, "on",2)) ? true : false ;
    } 
    else if (strncasecmp(line,"maxneckmoment",13)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      maxneckmoment = (double)lf;
    } 
    else if (strncasecmp(line,"minneckmoment",13)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      minneckmoment = (double)lf;
    } 
    else if (strncasecmp(line,"maxneckan",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      maxneckang = DEG2RAD((double)lf);
    } 
    else if (strncasecmp(line,"minneckan",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //minneckang = DEG2RAD((double)lf);
    } 
    else if (strncasecmp(line,"offside_kick_margin",9)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      //offside_kick_margin = (double)lf;
    } 
    else if (strncasecmp(line,"record_version",14)==0){
      // not needed
    } 
    else if (strncasecmp(line,"send_log",8)==0){
      // not needed
    } 
    else if (strncasecmp(line,"log_file",8)==0){
      // not needed
    } 
    else if (strncasecmp(line,"record_log",10)==0){
      // not needed
    }
    else if (strncasecmp(line,"log_times",9)==0){
      char log_times_string[128];
      n = sscanf(line, "%s %s", com, log_times_string ) ;
      //log_times = (!strncasecmp(log_times_string, "on",2)) ? true : false ;
    }
    else if (strncasecmp(line,"send_vi_step",8)==0){
      // not needed
    } 
    else if (strncasecmp(line,"say_coach_msg_size",16)==0){
      // not needed
    } 
    else if (strncasecmp(line,"say_coach_cnt_max",16)==0){
      // not needed
    } 
    else if (strncasecmp(line,"sense_body_step",10)==0){
      // not needed
    }
    else if (strncasecmp(line,"clang_win_size",14)==0){
      //n = sscanf(line, "%s %d", com, &clang_win_size ) ;
    }
    else if (strncasecmp(line,"clang_define_win",16)==0){
      //n = sscanf(line, "%s %d", com, &clang_define_win ) ;
    }
    else if (strncasecmp(line,"clang_meta_win",14)==0){
      //n = sscanf(line, "%s %d", com, &clang_meta_win ) ;
    }
    else if (strncasecmp(line,"clang_advice_win",16)==0){
      //n = sscanf(line, "%s %d", com, &clang_advice_win ) ;
    }
    else if (strncasecmp(line,"clang_info_win",14)==0){
      //n = sscanf(line, "%s %d", com, &clang_info_win ) ;
    }
    else if (strncasecmp(line,"clang_mess_delay",16)==0){
      //n = sscanf(line, "%s %d", com, &clang_mess_delay ) ;
    }
    else if (strncasecmp(line,"clang_mess_per_cycle",20)==0){
      //n = sscanf(line, "%s %d", com, &clang_mess_per_cycle ) ;
    }
    else if (strncasecmp(line,"tackle_dist",11)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      tackle_dist= double(lf);
    } 
    else if (strncasecmp(line,"tackle_back_dist",16)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      tackle_back_dist= double(lf);
    } 
    else if (strncasecmp(line,"tackle_width",12)==0){
      n = sscanf(line, "%s %lf", com, &lf ) ;
      tackle_width= double(lf);
    } 
    else if (strncasecmp(line,"tackle_exponent",15)==0){
      n = sscanf(line, "%s %d", com, &tackle_exponent ) ;
    }     
    else if (strncasecmp(line,"tackle_cycles",13)==0){
      n = sscanf(line, "%s %d", com, &tackle_cycles ) ;
    } 
    else {
      cerr << "\n Unkown option in server.conf! \n" << line << endl;
    } 
    // cout << " bei file einlesen " << line << endl;
   
  }
  // cout << " nach file einlesen " << endl;
  kickable_area = kickable_margin + ball_size + player_size;

  return true;
}
#endif
