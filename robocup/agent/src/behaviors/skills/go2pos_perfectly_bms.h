#ifndef _GO2POS_PERFECTLY_BMS_H_
#define _GO2POS_PERFECTLY_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "Vector.h"
#include "log_macros.h"
#include "basic_cmd_bms.h"
#include "globaldef.h"
#include "ws_info.h"
#include "neuro_go2pos_bms.h"
#include "face_ball_bms.h"
#include "search_ball_bms.h" 
#include "tools.h"

/**
 *  This behavior tries to go to a position in a perfekt manner,
 *  in such  a way, that it tries to position the player behind the ball
 *  on a line wit the opponents goal center.
 */

class Go2PosPerfectly : public BodyBehavior {
  static bool initialized;
  
  NeuroGo2Pos *ivpNeuroGo2Pos;
  FaceBall *ivpFaceBall;
  SearchBall *ivpSearchBall;
  BasicCmd *ivpBasicCmd;
  int init_state;
  bool done_pos;
  
  public:
  	const static double start_nearing_at;
  	const static double start_nearing_tolerance;
  	const static double min_nearing_angle;
  	const static double min_nearing_dist;
  	const static double speed_dash;
  	
  	void reset_state() {
  		done_pos = false;
  		init_state = 0;
  	};
  	bool done_positioning() {
  		return done_pos;
  	};
  	bool get_cmd(Cmd & cmd);
  	static bool init(char const * conf_file, int argc, char const* const* argv) {
  	if( initialized )
		return true;
	
	initialized = true && NeuroGo2Pos::init(conf_file, argc, argv)
					   && FaceBall::init(conf_file, argc, argv)
					   && SearchBall::init(conf_file, argc, argv)
					   && BasicCmd::init(conf_file, argc, argv);
	return initialized;
    };
  	Go2PosPerfectly();
  	virtual ~Go2PosPerfectly() {
  		if(ivpNeuroGo2Pos)
  			delete ivpNeuroGo2Pos;
  		if(ivpFaceBall)
  			delete ivpFaceBall;
  		if(ivpSearchBall)
  			delete ivpSearchBall;
  		if(ivpBasicCmd)
  			delete ivpBasicCmd;
  	};
  	
 };
#endif
