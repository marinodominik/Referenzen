#include "train_skills_bmc.h"

#include "ws_info.h"
#include "log_macros.h"
#include "math.h"

#include "../basics/wmoptions.h"

#define TOLERANCE 0.3
#define TRAINING_RANGE 20

bool TrainSkillsPlayer::initialized = false;

TrainSkillsPlayer::TrainSkillsPlayer() { 
  go2pos         = new NeuroGo2Pos();
  neurointercept = new NeuroIntercept();
  neurokick      = new NeuroKick();
}

TrainSkillsPlayer::~TrainSkillsPlayer() {
  delete go2pos;
  delete neurointercept;
  delete neurokick;
}

bool TrainSkillsPlayer::init( char const * conf_file, int argc, char const* const * argv )
{
    if( initialized ) return initialized;
    initialized =
            NeuroGo2Pos::init(    conf_file, argc, argv ) &&
            NeuroKick::init(      conf_file, argc, argv ) &&
            NeuroIntercept::init( conf_file, argc, argv );
    return initialized;
}

bool TrainSkillsPlayer::playon(Cmd & cmd) {
#if 0  // train neurogo2pos
  LOG_POL(0, << _2D << C2D(target.x, target.y, 1, "ff0000") );
  go2pos->set_target(target);
  if((go2pos->get_cmd(cmd) == false) || //no command was set -> target reached!
     (num_cycles_in_trial > MAX_CYCLES_PER_TRIAL)){
    num_cycles_in_trial = 0;
    do{
      target = WSinfo::me->pos + Vector((drand48() -.5)*2.0 * TRAINING_RANGE, 
					(drand48() -.5)*2.0 * TRAINING_RANGE);
    } while (WSinfo::me->pos.sqr_distance(target) < SQUARE(1.0));
  }       
  return true;
#endif

  if(!WSinfo::is_ball_kickable()) {
    if(neurointercept->get_cmd(cmd) == false){// at ball ?
    }
  } 
  else {
    Vector target;
    do{
      target = WSinfo::me->pos + Vector((drand48() -.5)*2.0 * TRAINING_RANGE, 
					(drand48() -.5)*2.0 * TRAINING_RANGE);
    } while (target.sqr_distance(Vector(0,0)) > SQUARE(30.0));
    neurokick->kick_to_pos_with_final_vel(0,target);
    //    neurokick->get_cmd(cmd);
    cmd.cmd_body.set_turn(0.);
  }

  return true;  
}


bool TrainSkillsPlayer::get_cmd(Cmd & cmd) {
  if(WMoptions::offline){ // no server connected -> go directly to training  
    neurointercept->get_cmd(cmd);
    return true;
  }

  switch(WSinfo::ws->play_mode){
  case PM_PlayOn:
    return playon(cmd);
    break;
  default:
#if 1
    if(WSinfo::me->pos.sqr_distance(Vector(-5.,0)) > SQUARE(1.)){
      cmd.cmd_body.set_moveto(-5.,0.);
      return true;
    }
#endif
    return playon(cmd);
    return true;
  }

  LOG_POL(0, << "TrainSkills: now in cycle : " << WSinfo::ws->time);

  //LOG_DEB(0, << _2D << C2D(WSinfo::me->pos.x, WSinfo::me->pos.y, 2, "ff0000") );

  return false;
}
