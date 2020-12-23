#include "sputplayer_bmc.h"

bool SputPlayer::initialized = false;

bool SputPlayer::init(char const * conf_file, int argc, char const* const* argv)
{
    if( initialized ) return initialized;

    initialized =
            NeuroKick::init(conf_file,argc,argv) &&
            NeuroGo2Pos::init(conf_file,argc,argv) &&
            OneStepKick::init(conf_file,argc,argv) &&
            OneOrTwoStepKick::init(conf_file,argc,argv) &&
            NeuroIntercept::init(conf_file,argc,argv) &&
            BasicCmd::init(conf_file,argc,argv) &&
            OneTwoHoldTurn::init(conf_file,argc,argv) &&
            DribbleStraight::init(conf_file,argc,argv) &&
            SearchBall::init(conf_file,argc,argv) &&
            FaceBall::init(conf_file,argc,argv) &&
            Selfpass::init(conf_file,argc,argv);

    return initialized;
}

SputPlayer::SputPlayer()
{
    neurokick       = new NeuroKick();
    go2pos          = new NeuroGo2Pos();
    onestepkick     = new OneStepKick();
    oneortwo        = new OneOrTwoStepKick();
    intercept       = new NeuroIntercept();
    basic           = new BasicCmd();
    holdturn        = new OneTwoHoldTurn();
    dribblestraight = new DribbleStraight();
    searchball      = new SearchBall();
    faceball        = new FaceBall();
    selfpass        = new Selfpass();

    flg = false;
}

SputPlayer::~SputPlayer()
{
    delete neurokick;
    delete go2pos;
    delete onestepkick;
    delete oneortwo;
    delete intercept;
    delete basic;
    delete searchball;
    delete faceball;
    delete dribblestraight;
}

bool SputPlayer::get_cmd(Cmd & cmd) {
  //LOG_POL(0,<<"goal last seen: "<<WSmemory::last_seen_to_point(HIS_GOAL_LEFT_CORNER));
  //LOG_POL(0,<<" dir last seen: "<<WSmemory::last_seen_in_dir(0));
  Vector dumvec;
  switch(WSinfo::ws->play_mode) {
  case PM_PlayOn:
#if 0
    if(!WSinfo::is_ball_kickable()) {
    //go2pos->set_target(WSinfo::ball->pos);
    //go2pos->get_cmd(cmd);
      std::cerr << "\n#"<<WSinfo::ws->time<<" Ball not kickable!";
      intercept->get_cmd(cmd);
    } else {
      Value dumspeed;
      int dumsteps,dumnr;
      Vector dumipos,dumop;
      if(selfpass->is_selfpass_safe(ANGLE(0),dumspeed,dumipos,dumsteps,dumop,dumnr)) {
	LOG_POL(0,<<"Passing to self");
	selfpass->get_cmd(cmd,ANGLE(0),dumspeed,dumipos);
      } else {	
	if(std::fabs(WSinfo::me->ang.get_value_mPI_pPI())>.1) {
	  LOG_POL(0,<<"Turning ["<<-WSinfo::me->ang.get_value()<<"]");
	  cmd.cmd_body.set_turn(-WSinfo::me->ang.get_value());
	} else {
	  if(dribblestraight->is_dribble_safe(1)) {
	    dribblestraight->get_cmd(cmd);
	  } else {
	    if(holdturn->is_holdturn_safe()) {
	      holdturn->get_cmd(cmd);
	    } else {
	      std::cerr << "\n#"<<WSinfo::ws->time<<"No move possible!";
	      cmd.cmd_body.set_turn(0);
	    }
	  }
	}
      }
    }
#else
    if(!WSinfo::is_ball_pos_valid()) {
      faceball->turn_to_ball();
      faceball->get_cmd(cmd);
      return true;
    }
    if(!WSinfo::is_ball_kickable()) {
      if(flg) {
	std::cerr << "\nCyc #"<<WSinfo::ws->time<<": Lost ball unexpectedly!";
	flg=false;
      }
      intercept->get_cmd(cmd);
      return true;
    }
    if(!holdturn->is_holdturn_safe()) {
      LOG_POL(0,<<"HoldTurn not safe, kicking ball away!");
      neurokick->kick_to_pos_with_initial_vel(.6,Vector(0,0));
      neurokick->get_cmd(cmd);
      flg=false;
      return true;
    }
    flg=true;
    LOG_POL(0,<<"HoldTurn SAFE!");
    holdturn->get_cmd(cmd);
    return true;
    

#endif
  break;
  default:
    ;
    //cmd.cmd_main.set_turn(1.57);
  }

  //LOG_DEB(0, << "now in cycle : " << WSinfo::ws->time);

  //LOG_DEB(0, << _2D << C2D(WSinfo::me->pos.x, WSinfo::me->pos.y, 2, "ff0000") );
  //if ( WSinfo::ws->time % 100 == 1 ) {
  //  cout << "\nnow in cycle " << WSinfo::ws->time << flush;
  //}

  if(!cmd.cmd_body.is_cmd_set()) cmd.cmd_body.set_turn(0);
  return true;

}
