#include "selfpass_bms.h"
#include "tools.h"

bool Selfpass::initialized= false;

#if 0
#define DBLOG_MOV(LLL,XXX) LOG_POL(LLL,<<"Selfpass: "<<XXX)
//#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#define DBLOG_DRAW(LLL,XXX)
#else
#define DBLOG_MOV(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#endif

/*************************************************************************/

/* Initialization */

bool Selfpass::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized; // only initialize once...
  initialized = (BasicCmd::init(conf_file, argc, argv) &&
          OneOrTwoStepKick::init(conf_file,argc,argv));
  if(!initialized) { cout<<"\nSelfpass behavior NOT initialized!!!"; }
  return initialized;
}

Selfpass::Selfpass(){
  basic_cmd = new BasicCmd;
  onetwostepkick = new OneOrTwoStepKick;
}

Selfpass::~Selfpass() {
  delete basic_cmd;
}

bool Selfpass::get_cmd(Cmd & cmd){
  return false;
}

bool Selfpass::get_cmd(Cmd & cmd, const ANGLE targetdir, const double speed, const Vector target){
PlayerSet opp=WSinfo::alive_opponents;
opp.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
return get_cmd(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,WSinfo::ball->pos,WSinfo::ball->vel,
                           opp[0]->pos,opp[0]->ang,cmd,targetdir,speed,target);
}

bool Selfpass::get_cmd(Vector my_pos,Vector my_vel,ANGLE my_ang,Vector ball_pos,Vector ball_vel,
Vector opp_pos,ANGLE opp_ang,Cmd & cmd, const ANGLE targetdir, const double speed, const Vector target){
  double speed1, speed2;

	// for cijat hack (see below)
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	ANGLE new_my_ang;

  if(speed >0){ // first command is a kick
    LOG_MOV(0," Selfpass: get cmd: 12stepkick "
	      <<" speed  "<<speed<<" to target "<<target);
    onetwostepkick->set_state(my_pos,my_vel,my_ang,ball_pos,ball_vel,
                                               opp_pos,opp_ang,0);
    onetwostepkick->kick_to_pos_with_initial_vel(speed,target);
    onetwostepkick->get_vel(speed1,speed2);
    if(fabs(speed1-speed)>0.1){
      LOG_MOV(0,"Get cmd HMM - onestepkick not possible !-> Trying 2 step kick");
    }
    onetwostepkick->get_cmd(cmd);
    return true;
  }
  // begin movement
  if(targetdir.diff(my_ang) >5/180.*PI){ // have to turn
    double moment = (targetdir-my_ang).get_value_mPI_pPI() *
      (1.0 + (WSinfo::me->inertia_moment * (my_vel.norm())));
    if (moment > 3.14) moment = 3.14;
    if (moment < -3.14) moment = -3.14;
    basic_cmd->set_turn(moment);
    DBLOG_MOV(0," get cmd "
	      <<" targetdir "<<RAD2DEG(targetdir.get_value())
	      <<" tmpdir "<<RAD2DEG(WSinfo::me->ang.get_value())
	      <<" computed moment "<<RAD2DEG(moment)
	      <<": Have to turn by "<<RAD2DEG((targetdir-WSinfo::me->ang).get_value_mPI_pPI()));
    basic_cmd->get_cmd(cmd);
    return true;
  }
  else{ // kicked, turned, now dashing
		//cijat: hack to avoid collisions BEGIN
		// (was: just dash 100)
		Cmd tmpCmd;
		int dash_pow=100;
		static const float minDistToBall = 1.35*(ServerOptions::player_size + ServerOptions::ball_size);
		float distToBall;
		bool isSafe = false;
		while(dash_pow>90){
			tmpCmd.cmd_body.unset_lock();
			basic_cmd->set_dash(dash_pow);
			basic_cmd->get_cmd(tmpCmd);
			Tools::model_cmd_main(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,WSinfo::ball->pos,WSinfo::ball->vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
			distToBall = (new_my_pos-new_ball_pos).norm();
			if(distToBall>minDistToBall) {
				isSafe = true;
				break;
			}
			dash_pow-=10;
		}
		if(!isSafe)
			return false;
    DBLOG_MOV(0,"get cmd: Dash  " <<dash_pow << " (hacked by cijat)");
    basic_cmd->set_dash(dash_pow);
		//cijat hack END
    basic_cmd->get_cmd(cmd);
    return true;
  }
  return false;
}

void Selfpass::determine_optime2react(const ANGLE targetdir, const int max_dashes){
#define MAX_STEPS 10

  Cmd tmp_cmd;
  Vector desired_ballpos_rel2me;
  desired_ballpos_rel2me.init_polar (0.7,targetdir.get_value()); // that's where the ball finally should be
  Vector tmp_pos = WSinfo::me->pos + desired_ballpos_rel2me;
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  Vector tmp_vel = WSinfo::me->vel;
  Vector tmp_ballvel = WSinfo::ball->vel;
  Vector tmp_ballpos = WSinfo::ball->pos;
  ANGLE tmp_ang = WSinfo::me->ang;
  Angle next_ang;
  const int cycles2kick = 1; // assumption: only one cycle is needed to kick.
  double opradius;
  int tmp_stamina = (int) WSinfo::me->stamina;
  int next_stamina;
  const int op_time2react = 0;

  int i=0;
  for (int idx = 0; idx <50; idx++) 
    optime2react[idx] = -1; // init

  while(i<max_dashes +3){
    //    DBLOG_MOV(0,"determine ops: step "<<i);
    // simulate player: 
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();
      
    if(i<cycles2kick){ // do kicks
      basic_cmd->set_turn(0);
      // do nothing, but simulate player movement (velocity)
      tmp_ballvel = WSinfo::ball->vel; // virtually hold the ball to test what happens, if ball just
      tmp_ballpos = WSinfo::ball->pos; // continues without being kicked
    }
    else if(targetdir.diff(tmp_ang) >5/180.*PI){ // have to turn
      double moment = (targetdir-tmp_ang).get_value_mPI_pPI() *
	(1.0 + (WSinfo::me->inertia_moment * (tmp_vel.norm())));
      if (moment > 3.14) moment = 3.14;
      if (moment < -3.14) moment = -3.14;
      basic_cmd->set_turn(moment);
      basic_cmd->get_cmd(tmp_cmd);
    }
    else{ // kicked, turned, now dashing
      int dash_power = 100;
      if(tmp_stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
	dash_power = (int)WSinfo::me->stamina_inc_max;
	DBLOG_MOV(2,"Reducing Stamina to "<<dash_power);
      }
      DBLOG_MOV(2,"step "<<i<<": Dash "<<dash_power);
      basic_cmd->set_dash(dash_power);
      basic_cmd->get_cmd(tmp_cmd);	
    }
      
    Tools::model_cmd_main(tmp_pos,tmp_vel,tmp_ang.get_value(),tmp_stamina, tmp_ballpos, tmp_ballvel,
			  tmp_cmd.cmd_body, 
			  next_pos,next_vel,next_ang,next_stamina, next_ballpos, next_ballvel);
    DBLOG_MOV(2,"step "<<i<<" new position: "<<next_pos<<" new angle "<<RAD2DEG(next_ang)
	      <<" new stamina "<<next_stamina);
    DBLOG_DRAW(0,C2D(next_pos.x,next_pos.y,0.1,"red"));
    DBLOG_DRAW(0,C2D(next_ballpos.x,next_ballpos.y,0.1,"grey"));
      
    // check opponents;
    PlayerSet pset = WSinfo::valid_opponents;

    //DBLOG_MOV(0,"num ops "<<pset.num);
    for (int idx = 0; idx < pset.num; idx++) {
      if(optime2react[idx]>=0)
	continue; // already set
      //DBLOG_MOV(0,"checking idx "<<idx);
      if(pset[idx]->number == WSinfo::ws->his_goalie_number){
	opradius = ServerOptions::catchable_area_l + (i+(1-op_time2react)) * ServerOptions::player_speed_max;
      }
      else
	opradius = pset[idx]->kick_radius + (i+(1-op_time2react)) * ServerOptions::player_speed_max;
      // ridi: Worst Case: max. vel.
      // refinement: if opponent has a bad body dir, compute acceleration of opponent
      //DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
      if(next_pos.distance(pset[idx]->pos) < opradius){
	DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	if(pset[idx]->ang.diff((next_pos-pset[idx]->pos).ARG()) >5./180.*PI){
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me but has to turn");
	  optime2react[idx] = 1; // opponent has to turn
	}
	else{
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me and must not turn");
	  optime2react[idx] = 0; 
	}
	if(WSinfo::his_goalie != NULL){
	  if(pset[idx] == WSinfo::his_goalie)
	    optime2react[idx] = 0; // take special care of goalie!	  
	}
      }  //op gets me
    }  // for all opponents...
    i++;
  } // while i< max_steps

  for (int idx = 0; idx <50; idx++) 
    if(optime2react[idx] == -1)  // reset all those w.no normal time2react
      optime2react[idx] = 0; 
}





bool Selfpass::is_selfpass_safe(const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
				Vector &attacking_op, int & op_number, const int max_dashes,
				double op_time2react){

  //op_time2react is the time that is assumed the opponents need to react. 0 is worst case, 1 assumes
  // that they need 1 cycle to react. This is already pretty aggressive and (nearly) safe

#define MAX_STEPS 10
#define OUTOF_PITCH_SAFETY 3
  //#define OUTOF_PITCH_SAFETY 10

  determine_optime2react(targetdir,max_dashes);

  Cmd tmp_cmd;
  Vector desired_ballpos_rel2me;
  desired_ballpos_rel2me.init_polar (0.7,targetdir.get_value()); // that's where the ball finally should be
  Vector safe_ipos = WSinfo::ball->pos;
  Vector tmp_pos = WSinfo::me->pos + desired_ballpos_rel2me;
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  Vector tmp_vel = WSinfo::me->vel;
  Vector tmp_ballvel = WSinfo::ball->vel;
  Vector tmp_ballpos = WSinfo::ball->pos;
  ANGLE tmp_ang = WSinfo::me->ang;
  Angle next_ang;
  double decay =1.;
  double summed_decay = 0;
  const int cycles2kick = 1; // assumption: only one cycle is needed to kick.
  bool op_gets_me = false;
  bool op_gets_ball = false;
  bool outof_pitch = false;
  double opradius;
  int safe_steps = 0;
  int safe_steps_without_kicking = 0;
  Vector safe_ipos_without_kicking = WSinfo::ball->pos;
  Vector attacker = Vector(52.0,0); // default: op goalie attacks
  Vector attacker_without_kicking = Vector(52.0,0); // default: op goalie attacks
  int number_of_dashes = 0;
  int safe_dashes_with_kick = 0;
  int safe_dashes_without_kick = 0;
  int tmp_stamina = (int) WSinfo::me->stamina;
  int next_stamina;
  int attacker_number = 0;
  int attacker_number_without_kicking = 0;
  double outof_pitch_safety = OUTOF_PITCH_SAFETY;

  // *  ridi 05: plausibilty check first!
  if(WSinfo::ball->pos.getX() + outof_pitch_safety> FIELD_BORDER_X){ // if Ball is already close to border of pitch
    if(fabs(targetdir.get_value_mPI_pPI() )<90/180. *PI){
      DBLOG_MOV(0,"Ball close to border. forbidden to proceed !");
      return false;
    }
  }

  if(WSinfo::ball->pos.getX() - outof_pitch_safety<- FIELD_BORDER_X){ // if Ball is already close to border of pitch
    if(fabs(targetdir.get_value_mPI_pPI() )>90/180. *PI){
      DBLOG_MOV(0,"Ball close to border. forbidden to proceed !");
      return false;
    }
  }

  if(WSinfo::ball->pos.getY() + outof_pitch_safety> FIELD_BORDER_Y){ // if Ball is already close to border of pitch
    if(targetdir.get_value_mPI_pPI()<0){
      DBLOG_MOV(0,"Ball close to border. forbidden to proceed !");
      return false;
    }
  }

  if(WSinfo::ball->pos.getY() - outof_pitch_safety<- FIELD_BORDER_Y){ // if Ball is already close to border of pitch
    if(targetdir.get_value_mPI_pPI()>0){
      DBLOG_MOV(0,"Ball close to border. forbidden to proceed !");
      return false;
    }
  }

  if(FIELD_BORDER_Y- fabs(WSinfo::ball->pos.getY()) < outof_pitch_safety) // if Ball is already close to border of pitch
    outof_pitch_safety = fabs(FIELD_BORDER_Y-WSinfo::ball->pos.getY());
  if(FIELD_BORDER_X- fabs(WSinfo::ball->pos.getX()) < outof_pitch_safety) // if Ball is already close to border of pitch
    outof_pitch_safety = fabs(FIELD_BORDER_X-WSinfo::ball->pos.getX());





  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);
  if(pset.num >0){ // set default values are set to closest opponent. This applies, if steps that I can go < max_steps
    attacker = pset[0]->pos;
    attacker_without_kicking = pset[0]->pos;
  }



  int i=0;

  while(i<MAX_STEPS && number_of_dashes <= max_dashes &&
	outof_pitch == false &&  (op_gets_me == false || op_gets_ball == false)){
      // simulate player: 
      tmp_cmd.cmd_body.unset_lock();
      tmp_cmd.cmd_body.unset_cmd();
      
      if(i<cycles2kick){ // do kicks
	basic_cmd->set_turn(0);
	// do nothing, but simulate player movement (velocity)
	tmp_ballvel = WSinfo::ball->vel; // virtually hold the ball to test what happens, if ball just
	tmp_ballpos = WSinfo::ball->pos; // continues without being kicked
      }
      else if(targetdir.diff(tmp_ang) >5/180.*PI){ // have to turn
	double moment = (targetdir-tmp_ang).get_value_mPI_pPI() *
	  (1.0 + (WSinfo::me->inertia_moment * (tmp_vel.norm())));
	if (moment > 3.14) moment = 3.14;
	if (moment < -3.14) moment = -3.14;
	basic_cmd->set_turn(moment);
	DBLOG_MOV(2,"step "<<i
		  <<" targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" tmpdir "<<RAD2DEG(tmp_ang.get_value())
		  <<" computed moment "<<RAD2DEG(moment)
		  <<": Have to turn by "<<RAD2DEG((targetdir-tmp_ang).get_value_mPI_pPI()));
	basic_cmd->get_cmd(tmp_cmd);
      }
      else{ // kicked, turned, now dashing
	int dash_power = 100;
	if(tmp_stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
	  dash_power = (int)WSinfo::me->stamina_inc_max;
	  DBLOG_MOV(2,"Reducing Stamina to "<<dash_power);
	}
	DBLOG_MOV(2,"step "<<i<<": Dash "<<dash_power);
	basic_cmd->set_dash(dash_power);
	basic_cmd->get_cmd(tmp_cmd);	
	number_of_dashes ++;
      }
      
      Tools::model_cmd_main(tmp_pos,tmp_vel,tmp_ang.get_value(),tmp_stamina, tmp_ballpos, tmp_ballvel,
			    tmp_cmd.cmd_body, 
			    next_pos,next_vel,next_ang,next_stamina, next_ballpos, next_ballvel);
      DBLOG_MOV(0,"step "<<i<<" new position: "<<next_pos<<" new angle "<<RAD2DEG(next_ang)
		<<" new stamina "<<next_stamina);
      DBLOG_DRAW(0,C2D(next_pos.x,next_pos.y,0.1,"red"));
      DBLOG_DRAW(0,C2D(next_ballpos.x,next_ballpos.y,0.1,"grey"));
      
      //      if(op_time2react >0.0){ // check ballpos against fast advancing op; not needed if optime2react = 1
      if(true){ // check this in any case!
	pset= WSinfo::valid_opponents;
	pset.keep_players_in_circle(next_ballpos, ServerOptions::player_speed_max +.2);
	if(pset.num >0){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" within speed range");
	}
	pset.keep_and_sort_closest_players_to_point(1,next_ballpos); // considr only close ops. for correct
	if(pset.num >0){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" close ");
	  if(Tools::is_ballpos_safe(pset[0],next_ballpos) == false){
	    DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" gets ball -> BALLpos not safe!");
	    op_gets_ball = true;
	    attacker_number_without_kicking = pset[0]->number;
	    attacker_without_kicking = pset[0]->pos;
	  } // not safe
	} // pset
      } //optime2reac <1

      if(Tools::is_position_in_pitch(next_pos, outof_pitch_safety) == false){ // ridi05: was next_pos
	outof_pitch = true;
	DBLOG_MOV(0,"Position out of PITCH step "<<i<<" new position: "<<next_pos<<" out of pitch. safety "<<outof_pitch_safety);
      }
      else {
	DBLOG_MOV(0,"BALL IS SAFE step "<<i<<" new position: "<<next_ballpos<<" safety "<<outof_pitch_safety);
      }

      
      // check opponents;
      pset = WSinfo::valid_opponents;

      for (int idx = 0; idx < pset.num; idx++) {
	//double individual_time2react = op_time2react;  //previous implementation before 30.6.03
	double individual_time2react = optime2react[idx];  //previous implementation before 30.6.03
	individual_time2react += op_time2react;  //for risky selfpasses

	//double opradius = ServerOptions::kickable_area + i * ServerOptions::player_speed_max;
	if(pset[idx]->number == WSinfo::ws->his_goalie_number){
	  opradius = ServerOptions::catchable_area_l + (i+(1-individual_time2react)) * ServerOptions::player_speed_max; 
	  //	  if(individual_time2react >= 1){ // add an extra to the goalie, otherwise the ball is lost.
	  if(0){ // add an extra to the goalie, otherwise the ball is lost.
	    opradius += ServerOptions::player_speed_max; 
	  }
	}
	else
	  opradius = pset[idx]->kick_radius + (i+(1-individual_time2react)) * ServerOptions::player_speed_max; 
	// ridi: Worst Case: max. vel.
	// refinement: if opponent has a bad body dir, compute acceleration of opponent
	//DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	if(next_pos.distance(pset[idx]->pos) < opradius){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me his time 2 react"
		    <<optime2react[idx]);
	  if(op_gets_me == false){ // only for the first time
	    DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	    attacker_number = pset[idx]->number;
	    attacker = pset[idx]->pos;
	    DBLOG_DRAW(0,C2D(safe_ipos.x,safe_ipos.y,0.3,"orange"));
	  }
	  op_gets_me = true;
	}
	//	if(next_ballpos.distance(pset[idx]->pos) < (opradius -ServerOptions::player_speed_max )){
	if(tmp_ballpos.distance(pset[idx]->pos) < opradius){
	  // tricky: this simulation is one time step behind, since no kicking is considered
	  // but it is done in parallel
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets ball");
	  if(op_gets_ball == false){ 
	    DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius+.2,"blue"));
	    attacker_number_without_kicking = pset[idx]->number;
	    attacker_without_kicking = pset[idx]->pos;
	    DBLOG_DRAW(0,C2D(safe_ipos_without_kicking.x,safe_ipos_without_kicking.y,0.3,"#999999"));
	  }
	  op_gets_ball = true;
	}      
      } // for all players
      if(op_gets_me == false && outof_pitch == false){
	safe_ipos = next_pos;  // next_pos is safe
	safe_steps = i;
	safe_dashes_with_kick = number_of_dashes;
	summed_decay += decay;
      }
      Vector corrected_playerpos = next_pos -   desired_ballpos_rel2me;
      DBLOG_DRAW(0,C2D(corrected_playerpos.x,corrected_playerpos.y,1.2,"grey"));
#if 0
      if(op_gets_ball == false && 
	 ((next_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.2)) ||
	  (Tools::get_abs_angle((next_ballpos - corrected_playerpos).arg() - targetdir.get_value()) 
	   < 90/180.*PI && (next_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.15)))
	  )){
#endif
      if(op_gets_ball == false && 
	 ((tmp_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.2)) ||
	  (Tools::get_abs_angle((tmp_ballpos - corrected_playerpos).arg() - targetdir.get_value()) 
	   < 90/180.*PI && (tmp_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.15)))
	  )){
	// tricky: Without kick simultation is one step behind, therefore tmp_ballpos is used
	// opponent hasnt got the ball so far, and it is close to me!
	// and its in an ok.position
	DBLOG_DRAW(0,C2D(tmp_ballpos.x,tmp_ballpos.y,0.1,"black"));
	safe_ipos_without_kicking = next_pos;  // next_pos is safe
	safe_steps_without_kicking = i-1; // this simulation is one step behind
	safe_dashes_without_kick = number_of_dashes;
      }
      tmp_pos = next_pos;
      tmp_vel = next_vel;
      tmp_ballpos = next_ballpos;
      tmp_ballvel = next_ballvel;
      tmp_ang = ANGLE(next_ang);
      tmp_stamina = next_stamina;
      //DBLOG_DRAW(0,C2D(safe_ipos.x,safe_ipos.y,0.3,"orange"));
      //DBLOG_DRAW(0,C2D(safe_ipos_without_kicking.x,safe_ipos_without_kicking.y,0.3,"#aaaaaa"));
      decay *= ServerOptions::ball_decay;
      i++;
  };  // while

  double dist_with_kick = (WSinfo::ball->pos - safe_ipos).norm();
  //  double dist_without_kick = (WSinfo::ball->pos - safe_ipos_without_kicking).norm();

  // first, compute if one step kick is possible as assumed by routine above.
  double speed1, speed2;
  speed = dist_with_kick/summed_decay;
  onetwostepkick->reset_state(); // use current ws-state
  onetwostepkick->kick_to_pos_with_initial_vel(speed,safe_ipos);
  onetwostepkick->get_vel(speed1,speed2);
  if(fabs(speed1-speed)>0.1){
    DBLOG_MOV(0,"HMM - onestepkick not possible !-> trying anyhow, since ball remains in kickrange");
    if(safe_dashes_with_kick <2) // in this case it makes no sense to try
      safe_dashes_with_kick = 0;
    if(fabs(speed2-speed)>0.1){
      DBLOG_MOV(0,"OUCH - even two step kick not possible -> give up with selfpass");
      safe_dashes_with_kick = 0;
    }
  }

  if(safe_dashes_without_kick>0){ // I can advance without kicking
    if(safe_dashes_without_kick >= safe_dashes_with_kick){
      DBLOG_MOV(0,"WOW! I can do a successfulselfpass without kicking, steps: "<<safe_steps_without_kicking
		<<" dashes :"<<safe_dashes_without_kick<<" dashes with kicking "
		<<safe_dashes_with_kick);
      ipos = safe_ipos_without_kicking;
      steps = safe_steps_without_kicking;
      attacking_op = attacker_without_kicking;
      op_number = attacker_number_without_kicking;
      speed = 0;
      return true;
    }
  }

  if(safe_dashes_with_kick>0){ // I advance 
    ipos = safe_ipos;
    steps = safe_steps;
    attacking_op = attacker;
    op_number = attacker_number;

    DBLOG_MOV(0,"Selfpass OK "
	      <<" steps "<< steps<<" speed "<<speed<<" dir "<<RAD2DEG(targetdir.get_value())
	      <<" dashes :"<<safe_dashes_with_kick<<" dashes without kicking "
	      <<safe_dashes_without_kick);
    return true;
  }

  DBLOG_MOV(0," selfpass in dir "<<RAD2DEG(targetdir.get_value())<<" not possible");
  speed = 0;
  ipos = WSinfo::ball->pos;
  steps = safe_steps;
  return false;
}




bool Selfpass::is_selfpass_safe( const AState & state, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
				Vector &attacking_op, int & op_number, const int max_dashes,
				double op_time2react ) {

  AObject me = state.my_team[state.my_idx];

  //op_time2react is the time that is assumed the opponents need to react. 0 is worst case, 1 assumes
  // that they need 1 cycle to react. This is already pretty aggressive and (nearly) safe

#define MAX_STEPS 10
#define OUTOF_PITCH_SAFETY 3
  //#define OUTOF_PITCH_SAFETY 10

  determine_optime2react(state, targetdir,max_dashes);

  Cmd tmp_cmd;
  Vector desired_ballpos_rel2me;
  desired_ballpos_rel2me.init_polar (0.7,targetdir.get_value()); // that's where the ball finally should be
  Vector safe_ipos = state.ball.pos;
  Vector tmp_pos = me.pos + desired_ballpos_rel2me;
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  Vector tmp_vel = me.vel;
  Vector tmp_ballvel = state.ball.vel;
  Vector tmp_ballpos = state.ball.pos;
  ANGLE tmp_ang = ANGLE(me.body_angle);
  Angle next_ang;
  double decay =1.;
  double summed_decay = 0;
  const int cycles2kick = 1; // assumption: only one cycle is needed to kick.
  bool op_gets_me = false;
  bool op_gets_ball = false;
  bool outof_pitch = false;
  double opradius;
  int safe_steps = 0;
  int safe_steps_without_kicking = 0;
  Vector safe_ipos_without_kicking = state.ball.pos;
  Vector attacker = Vector(52.0,0); // default: op goalie attacks
  Vector attacker_without_kicking = Vector(52.0,0); // default: op goalie attacks
  int number_of_dashes = 0;
  int safe_dashes_with_kick = 0;
  int safe_dashes_without_kick = 0;
  int tmp_stamina = (int) me.stamina;
  int next_stamina;
  int attacker_number = 0;
  int attacker_number_without_kicking = 0;
  double outof_pitch_safety = OUTOF_PITCH_SAFETY;

  if(FIELD_BORDER_Y- fabs(state.ball.pos.getY()) < outof_pitch_safety) // if Ball is already close to border of pitch
    outof_pitch_safety = fabs(FIELD_BORDER_Y-state.ball.pos.getY());
  if(FIELD_BORDER_X- fabs(state.ball.pos.getX()) < outof_pitch_safety) // if Ball is already close to border of pitch
    outof_pitch_safety = fabs(FIELD_BORDER_X-state.ball.pos.getX());





  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, me.pos);
  if(pset.num >0){ // set default values are set to closest opponent. This applies, if steps that I can go < max_steps
    attacker = pset[0]->pos;
    attacker_without_kicking = pset[0]->pos;
  }



  int i=0;

  while(i<MAX_STEPS && number_of_dashes <= max_dashes &&
	outof_pitch == false &&  (op_gets_me == false || op_gets_ball == false)){
      // simulate player: 
      tmp_cmd.cmd_body.unset_lock();
      tmp_cmd.cmd_body.unset_cmd();
      
      if(i<cycles2kick){ // do kicks
	basic_cmd->set_turn(0);
	// do nothing, but simulate player movement (velocity)
	tmp_ballvel = state.ball.vel; // virtually hold the ball to test what happens, if ball just
	tmp_ballpos = state.ball.pos; // continues without being kicked
      }
      else if (targetdir.diff(tmp_ang) >5/180.*PI) { // have to turn
	double moment = (targetdir-tmp_ang).get_value_mPI_pPI() *
	  (1.0 + (WSinfo::me->inertia_moment * (tmp_vel.norm())));
	if (moment > 3.14) moment = 3.14;
	if (moment < -3.14) moment = -3.14;
	basic_cmd->set_turn(moment);
	DBLOG_MOV(2,"step "<<i
		  <<" targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" tmpdir "<<RAD2DEG(tmp_ang.get_value())
		  <<" computed moment "<<RAD2DEG(moment)
		  <<": Have to turn by "<<RAD2DEG((targetdir-tmp_ang).get_value_mPI_pPI()));
	basic_cmd->get_cmd(tmp_cmd);
      }
      else{ // kicked, turned, now dashing
	int dash_power = 100;
	if(tmp_stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
	  dash_power = (int)WSinfo::me->stamina_inc_max;
	  DBLOG_MOV(2,"Reducing Stamina to "<<dash_power);
	}
	DBLOG_MOV(2,"step "<<i<<": Dash "<<dash_power);
	basic_cmd->set_dash(dash_power);
	basic_cmd->get_cmd(tmp_cmd);	
	number_of_dashes ++;
      }
      
      Tools::model_cmd_main(tmp_pos,tmp_vel,tmp_ang.get_value(),tmp_stamina, tmp_ballpos, tmp_ballvel,
			    tmp_cmd.cmd_body, 
			    next_pos,next_vel,next_ang,next_stamina, next_ballpos, next_ballvel);
      DBLOG_MOV(2,"step "<<i<<" new position: "<<next_pos<<" new angle "<<RAD2DEG(next_ang)
		<<" new stamina "<<next_stamina);
      DBLOG_DRAW(0,C2D(next_pos.x,next_pos.y,0.1,"red"));
      DBLOG_DRAW(0,C2D(next_ballpos.x,next_ballpos.y,0.1,"grey"));
      
      //      if(op_time2react >0.0){ // check ballpos against fast advancing op; not needed if optime2react = 1
      if(true){ // check this in any case!
	pset= WSinfo::valid_opponents;
	pset.keep_players_in_circle(next_ballpos, ServerOptions::player_speed_max +.2);
	if(pset.num >0){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" within speed range");
	}
	pset.keep_and_sort_closest_players_to_point(1,next_ballpos); // considr only close ops. for correct
	if(pset.num >0){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" close ");
	  if(Tools::is_ballpos_safe(pset[0],next_ballpos) == false){
	    DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[0]->number<<" gets ball -> BALLpos not safe!");
	    op_gets_ball = true;
	    attacker_number_without_kicking = pset[0]->number;
	    attacker_without_kicking = pset[0]->pos;
	  } // not safe
	} // pset
      } //optime2reac <1

      if(Tools::is_position_in_pitch(next_pos, outof_pitch_safety) == false){
	outof_pitch = true;
	DBLOG_MOV(0,"step "<<i<<" new position: "<<next_pos<<" out of pitch ");
      }
      
      // check opponents;
      pset = WSinfo::valid_opponents;

      for (int idx = 0; idx < pset.num; idx++) {
	//double individual_time2react = op_time2react;  //previous implementation before 30.6.03
	double individual_time2react = optime2react[idx];  //previous implementation before 30.6.03
	individual_time2react += op_time2react;  //for risky selfpasses

	//double opradius = ServerOptions::kickable_area + i * ServerOptions::player_speed_max;
	if(pset[idx]->number == WSinfo::ws->his_goalie_number){
	  opradius = ServerOptions::catchable_area_l + (i+(1-individual_time2react)) * ServerOptions::player_speed_max; 
	  //	  if(individual_time2react >= 1){ // add an extra to the goalie, otherwise the ball is lost.
	  if(0){ // add an extra to the goalie, otherwise the ball is lost.
	    opradius += ServerOptions::player_speed_max; 
	  }
	}
	else
	  opradius = pset[idx]->kick_radius + (i+(1-individual_time2react)) * ServerOptions::player_speed_max; 
	// ridi: Worst Case: max. vel.
	// refinement: if opponent has a bad body dir, compute acceleration of opponent
	//DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	if(next_pos.distance(pset[idx]->pos) < opradius){
	  DBLOG_MOV(0,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me his time 2 react"
		    <<optime2react[idx]);
	  if(op_gets_me == false){ // only for the first time
	    DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	    attacker_number = pset[idx]->number;
	    attacker = pset[idx]->pos;
	    DBLOG_DRAW(0,C2D(safe_ipos.x,safe_ipos.y,0.3,"orange"));
	  }
	  op_gets_me = true;
	}
	//	if(next_ballpos.distance(pset[idx]->pos) < (opradius -ServerOptions::player_speed_max )){
	if(tmp_ballpos.distance(pset[idx]->pos) < opradius){
	  // tricky: this simulation is one time step behind, since no kicking is considered
	  // but it is done in parallel
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets ball");
	  if(op_gets_ball == false){ 
	    DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius+.2,"blue"));
	    attacker_number_without_kicking = pset[idx]->number;
	    attacker_without_kicking = pset[idx]->pos;
	    DBLOG_DRAW(0,C2D(safe_ipos_without_kicking.x,safe_ipos_without_kicking.y,0.3,"#999999"));
	  }
	  op_gets_ball = true;
	}      
      } // for all players
      if(op_gets_me == false && outof_pitch == false){
	safe_ipos = next_pos;  // next_pos is safe
	safe_steps = i;
	safe_dashes_with_kick = number_of_dashes;
	summed_decay += decay;
      }
      Vector corrected_playerpos = next_pos -   desired_ballpos_rel2me;
      DBLOG_DRAW(0,C2D(corrected_playerpos.x,corrected_playerpos.y,1.2,"grey"));
#if 0
      if(op_gets_ball == false && 
	 ((next_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.2)) ||
	  (Tools::get_abs_angle((next_ballpos - corrected_playerpos).arg() - targetdir.get_value()) 
	   < 90/180.*PI && (next_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.15)))
	  )){
#endif
      if(op_gets_ball == false && 
	 ((tmp_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.2)) ||
	  (Tools::get_abs_angle((tmp_ballpos - corrected_playerpos).arg() - targetdir.get_value()) 
	   < 90/180.*PI && (tmp_ballpos.distance(corrected_playerpos) < (WSinfo::me->kick_radius - 0.15)))
	  )){
	// tricky: Without kick simultation is one step behind, therefore tmp_ballpos is used
	// opponent hasnt got the ball so far, and it is close to me!
	// and its in an ok.position
	DBLOG_DRAW(0,C2D(tmp_ballpos.x,tmp_ballpos.y,0.1,"black"));
	safe_ipos_without_kicking = next_pos;  // next_pos is safe
	safe_steps_without_kicking = i-1; // this simulation is one step behind
	safe_dashes_without_kick = number_of_dashes;
      }
      tmp_pos = next_pos;
      tmp_vel = next_vel;
      tmp_ballpos = next_ballpos;
      tmp_ballvel = next_ballvel;
      tmp_ang = ANGLE(next_ang);
      tmp_stamina = next_stamina;
      //DBLOG_DRAW(0,C2D(safe_ipos.x,safe_ipos.y,0.3,"orange"));
      //DBLOG_DRAW(0,C2D(safe_ipos_without_kicking.x,safe_ipos_without_kicking.y,0.3,"#aaaaaa"));
      decay *= ServerOptions::ball_decay;
      i++;
  };  // while

  double dist_with_kick = (state.ball.pos - safe_ipos).norm();
  //  double dist_without_kick = (state.ball.pos - safe_ipos_without_kicking).norm();

  // first, compute if one step kick is possible as assumed by routine above.
  double speed1, speed2;
  speed = dist_with_kick/summed_decay;

  onetwostepkick->set_state(state);
  onetwostepkick->kick_to_pos_with_initial_vel(speed,safe_ipos);
  onetwostepkick->get_vel(speed1,speed2);
  onetwostepkick->reset_state(); // use current ws-state
  if(fabs(speed1-speed)>0.1){
    DBLOG_MOV(0,"HMM - onestepkick not possible !-> trying anyhow, since ball remains in kickrange");
    if(safe_dashes_with_kick <2) // in this case it makes no sense to try
      safe_dashes_with_kick = 0;
    if(fabs(speed2-speed)>0.1){
      DBLOG_MOV(0,"OUCH - even two step kick not possible -> give up with selfpass");
      safe_dashes_with_kick = 0;
    }
  }

  if(safe_dashes_without_kick>0){ // I can advance without kicking
    if(safe_dashes_without_kick >= safe_dashes_with_kick){
      DBLOG_MOV(0,"WOW! I can do a successfulselfpass without kicking, steps: "<<safe_steps_without_kicking
		<<" dashes :"<<safe_dashes_without_kick<<" dashes with kicking "
		<<safe_dashes_with_kick);
      ipos = safe_ipos_without_kicking;
      steps = safe_steps_without_kicking;
      attacking_op = attacker_without_kicking;
      op_number = attacker_number_without_kicking;
      speed = 0;
      return true;
    }
  }

  if(safe_dashes_with_kick>0){ // I advance 
    ipos = safe_ipos;
    steps = safe_steps;
    attacking_op = attacker;
    op_number = attacker_number;

    DBLOG_MOV(0,"Selfpass OK "
	      <<" steps "<< steps<<" speed "<<speed<<" dir "<<RAD2DEG(targetdir.get_value())
	      <<" dashes :"<<safe_dashes_with_kick<<" dashes without kicking "
	      <<safe_dashes_without_kick);
    return true;
  }

  DBLOG_MOV(0," selfpass in dir "<<RAD2DEG(targetdir.get_value())<<" not possible");
  speed = 0;
  ipos = state.ball.pos;
  steps = safe_steps;
  return false;
}



void Selfpass::determine_optime2react( const AState & state, const ANGLE targetdir, const int max_dashes ) {
#define MAX_STEPS 10

  AObject me = state.my_team[state.my_idx];

  Cmd tmp_cmd;
  Vector desired_ballpos_rel2me;
  desired_ballpos_rel2me.init_polar (0.7,targetdir.get_value()); // that's where the ball finally should be
  Vector tmp_pos = me.pos + desired_ballpos_rel2me;
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  Vector tmp_vel = me.vel;
  Vector tmp_ballvel = state.ball.vel;
  Vector tmp_ballpos = state.ball.pos;
  ANGLE tmp_ang = ANGLE(me.body_angle);
  Angle next_ang;
  const int cycles2kick = 1; // assumption: only one cycle is needed to kick.
  double opradius;
  int tmp_stamina = (int) me.stamina;
  int next_stamina;
  const int op_time2react = 0;

  int i=0;
  for (int idx = 0; idx <50; idx++) 
    optime2react[idx] = -1; // init

  while(i<max_dashes +3){
    //    DBLOG_MOV(0,"determine ops: step "<<i);
    // simulate player: 
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();
      
    if(i<cycles2kick){ // do kicks
      basic_cmd->set_turn(0);
      // do nothing, but simulate player movement (velocity)
      tmp_ballvel = state.ball.vel; // virtually hold the ball to test what happens, if ball just
      tmp_ballpos = state.ball.pos; // continues without being kicked
    }
    else if(targetdir.diff(tmp_ang) >5/180.*PI){ // have to turn
      double moment = (targetdir-tmp_ang).get_value_mPI_pPI() *
	(1.0 + (WSinfo::me->inertia_moment * (tmp_vel.norm())));
      if (moment > 3.14) moment = 3.14;
      if (moment < -3.14) moment = -3.14;
      basic_cmd->set_turn(moment);
      basic_cmd->get_cmd(tmp_cmd);
    }
    else{ // kicked, turned, now dashing
      int dash_power = 100;
      if(tmp_stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
	dash_power = (int)WSinfo::me->stamina_inc_max;
	DBLOG_MOV(2,"Reducing Stamina to "<<dash_power);
      }
      DBLOG_MOV(2,"step "<<i<<": Dash "<<dash_power);
      basic_cmd->set_dash(dash_power);
      basic_cmd->get_cmd(tmp_cmd);	
    }
      
    Tools::model_cmd_main(tmp_pos,tmp_vel,tmp_ang.get_value(),tmp_stamina, tmp_ballpos, tmp_ballvel,
			  tmp_cmd.cmd_body, 
			  next_pos,next_vel,next_ang,next_stamina, next_ballpos, next_ballvel);
    DBLOG_MOV(2,"step "<<i<<" new position: "<<next_pos<<" new angle "<<RAD2DEG(next_ang)
	      <<" new stamina "<<next_stamina);
    DBLOG_DRAW(0,C2D(next_pos.x,next_pos.y,0.1,"red"));
    DBLOG_DRAW(0,C2D(next_ballpos.x,next_ballpos.y,0.1,"grey"));
      
    // check opponents;
    PlayerSet pset = WSinfo::valid_opponents;

    //DBLOG_MOV(0,"num ops "<<pset.num);
    for (int idx = 0; idx < pset.num; idx++) {
      if(optime2react[idx]>=0)
	continue; // already set
      //DBLOG_MOV(0,"checking idx "<<idx);
      if(pset[idx]->number == WSinfo::ws->his_goalie_number){
	opradius = ServerOptions::catchable_area_l + (i+(1-op_time2react)) * ServerOptions::player_speed_max;
      }
      else
	opradius = pset[idx]->kick_radius + (i+(1-op_time2react)) * ServerOptions::player_speed_max;
      // ridi: Worst Case: max. vel.
      // refinement: if opponent has a bad body dir, compute acceleration of opponent
      //DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
      if(next_pos.distance(pset[idx]->pos) < opradius){
	DBLOG_DRAW(0,C2D(pset[idx]->pos.x,pset[idx]->pos.y,opradius,"magenta"));
	if(pset[idx]->ang.diff((next_pos-pset[idx]->pos).ARG()) >5./180.*PI){
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me but has to turn");
	  optime2react[idx] = 1; // opponent has to turn
	}
	else{
	  DBLOG_MOV(2,"step "<<i<<" Opponent "<<pset[idx]->number<<" gets me and must not turn");
	  optime2react[idx] = 0; 
	}
	if(WSinfo::his_goalie != NULL){
	  if(pset[idx] == WSinfo::his_goalie)
	    optime2react[idx] = 0; // take special care of goalie!	  
	}
      }  //op gets me
    }  // for all opponents...
    i++;
  } // while i< max_steps

  for (int idx = 0; idx <50; idx++) 
    if(optime2react[idx] == -1)  // reset all those w.no normal time2react
      optime2react[idx] = 0; 
}
