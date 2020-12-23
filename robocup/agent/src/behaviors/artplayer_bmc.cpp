#include "artplayer_bmc.h"

#include "ws_info.h"
#include "log_macros.h"
#include "tools.h"

bool ArtPlayer::initialized=false;

//#include "formation.h" //debug
//Formation433 formation;

#include "intercept2.h"
#include "skills/one_step_kick_bms.h"

bool ArtPlayer::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true;
  initialized = true;
    
//  formation.init(CommandLineOptions::formations_conf,0,0);

  return (
	  NeuroGo2Pos::init(conf_file,argc,argv) &&
	  InterceptBall::init(conf_file,argc,argv) &&
	  OneStepKick::init(conf_file,argc,argv)
	  );
}

ArtPlayer::ArtPlayer() {
  go2pos = new NeuroGo2Pos;
  intercept = new InterceptBall;
}

ArtPlayer::~ArtPlayer() {
  delete go2pos;
  delete intercept;
}

bool ArtPlayer::get_cmd(Cmd & cmd) {
  //std::cout << "\nid= [" << id() << "]";

  Vector target(FIELD_BORDER_X-16,20);
  Intercept2 inter;
  PPlayer me= WSinfo::me;
  Vector pos= me->pos;
  ANGLE ang= me->ang;
  Vector tmp;
  int responsible_player;
  int time;
  bool use_dash= false;
  LOG_DEB(0, << "play_mode= " << play_mode_str(WSinfo::ws->play_mode));
  switch(WSinfo::ws->play_mode) {
  case PM_his_BeforePenaltyKick:
    cmd.cmd_body.set_turn(0);
    break;
  case PM_his_PenaltyKick:
    //cmd.cmd_main.set_turn(0); break; //debug
    if ( ClientOptions::consider_goalie )
      pos= Vector(- 50,-5);
    else
      pos= Vector( +1, WSinfo::me->number - 6 );
    if ( use_dash || ClientOptions::consider_goalie )
      inter.time_to_target( pos, 1.0, WSinfo::me, time, cmd.cmd_body); // this corresponds to go2pos
    break;
  case PM_my_BeforePenaltyKick:
    //cmd.cmd_main.set_turn(0); break;
  case PM_my_PenaltyKick:
    responsible_player= (WSinfo::ws->penalty_count+1) % NUM_PLAYERS + 1;
    //responsible_player= WSinfo::ws->penalty_count;
    if ( responsible_player != WSinfo::me->number ) {
      if ( ClientOptions::consider_goalie )
	pos= Vector( 54,-25);
      else
	pos= Vector( -1, WSinfo::me->number - 6 );
      if ( use_dash )
        inter.time_to_target( pos, 1.0, WSinfo::me, time, cmd.cmd_body); // this corresponds to go2pos
      else
        cmd.cmd_body.set_turn(0);

      break;
    }
    //no break here
  case PM_PlayOn:
//    target= formation.get_grid_pos(WSinfo::me->number);
    //target= formation.get_fine_pos(WSinfo::me->number);
    //target= DeltaPositioning::get_position(WSinfo::me->number);
    target= Vector(-FIELD_BORDER_X+16,20);
    if ( WSinfo::ws->time > 100 )
      target= Vector(FIELD_BORDER_X-16,20);
    go2pos->set_target(target);

    if ( WSinfo::is_ball_kickable() && WSinfo::ws->play_mode == PM_my_PenaltyKick ) {
      OneStepKick kick;
      kick.kick_to_pos_with_max_vel(Vector(52.0,0));
      kick.get_cmd(cmd);
      return true;
    }
    //inter.conv_to_canonical_form(Vector(0,0),pos,vel,ang);
    //inter.conv_to_canonical_form(target,pos,vel,ang);
    inter.time_to_target( WSinfo::ball->pos, 0.8, WSinfo::me, time, cmd.cmd_body); // this corresponds to go2pos
    //inter.intercept( WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::me, time, cmd.cmd_main);
    LOG_DEB(0, << "Rusulting time= " << time );
    return (time == 0);

    //std::cout << " pos= " << pos;
    return go2pos->get_cmd(cmd);
    break;
  default:
    cmd.cmd_body.set_turn(1.57);
  }

  //LOG_DEB(0, << "now in cycle : " << WSinfo::ws->time);

  //LOG_DEB(0, << _2D << C2D(WSinfo::me->pos.x, WSinfo::me->pos.y, 2, "ff0000") );
  //if ( WSinfo::ws->time % 100 == 1 ) {
  //  cout << "\nnow in cycle " << WSinfo::ws->time << flush;
  //}

  return true;

}
