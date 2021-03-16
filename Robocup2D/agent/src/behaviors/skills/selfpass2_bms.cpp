#include "selfpass2_bms.h"
#include "ws_memory.h"
#include "tools.h"

bool Selfpass2::initialized= false;

#if 1
#define DBLOG_MOV(LLL,XXX) LOG_POL(LLL,<<"Selfpass2: "<<XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
//#define DBLOG_DRAW(LLL,XXX)
#else
#define DBLOG_MOV(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#endif

#if 1
#define MRLOG_MOV(LLL,XXX) LOG_POL(LLL,<<"Selfpass2: "<<XXX)
#define MRLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
//#define DBLOG_DRAW(LLL,XXX)
#else
#define MRLOG_MOV(LLL,XXX)
#define MRLOG_DRAW(LLL,XXX)
#endif


#define BODYDIR_TOLERANCE 5.0   // allowed tolerance before turning into target direction
#define MIN_DIST_2_BORDER 4.  // keep away from border
#define SAFETY_MARGIN 0.2  // want to have the ball safe!

#define MINIMAL_SELFPASS_RISKLEVEL   0  //TG09
#define MAXIMAL_SELFPASS_RISKLEVEL   3//5  //TG09
#define MAXIMAL_SELFPASS_RISKLEVEL_CAREFUL 1//2 //TG16

/*************************************************************************/
int  Selfpass2::cvSelfpassRiskLevel = 1;//TG09
long Selfpass2::cvLastSelfpassRiskLevelChange = -1;

/* Initialization */

bool Selfpass2::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized; // only initialize once...
  initialized = (BasicCmd::init(conf_file, argc, argv) &&
          OneOrTwoStepKick::init(conf_file,argc,argv));
  if(!initialized) { cout<<"\nSelfpass2 behavior NOT initialized!!!"; }
  return initialized;
}

Selfpass2::Selfpass2(){
  basic_cmd = new BasicCmd;
  onetwostepkick = new OneOrTwoStepKick;
}

Selfpass2::~Selfpass2() {
  delete basic_cmd;
  delete onetwostepkick;
}

bool Selfpass2::get_cmd(Cmd & cmd){

  DBLOG_MOV(0,"get_cmd: No precomputation available! Call is_safe(targetdir) first!!!");
  return false;
}

/*
bool Selfpass2::get_cmd(Cmd & cmd){

  if(simulation_table[0].valid_at != WSinfo::ws->time){
    DBLOG_MOV(0,"get_cmd: No precomputation available! Call is_safe(targetdir) first!!!");
    return false;
  }
  cmd = simulation_table[0].cmd;
  return true;
}
*/

bool Selfpass2::get_cmd(Cmd & cmd, const ANGLE targetdir, const Vector targetpos, const double kickspeed){

  MRLOG_MOV(0,"get_cmd: targetdir: "<<RAD2DEG(targetdir.get_value())<<" targetpos "<<targetpos
	    <<" kickspeed "<<kickspeed);

	    
  MRLOG_DRAW(0,VC2D(targetpos,0.3,"orange"));

  if(kickspeed >0){ // want to kick -> compute appropriate onestep kick
    double speed1, speed2;
    onetwostepkick->reset_state(); // use current ws-state
    onetwostepkick->kick_to_pos_with_initial_vel(kickspeed,targetpos);
    //  onetwostepkick->kick_in_dir_with_initial_vel(kickspeed,targetdir);
    onetwostepkick->get_vel(speed1,speed2);
    if(fabs(speed1-kickspeed)<=0.1){
      // everything's fine!
      onetwostepkick->get_cmd(cmd);
      return true;
    }
    MRLOG_MOV(0,"get_cmd: PROBLEM. Should kick, but kick does not work. CALL ridi!!!");
    return false;
  }

  if(kickspeed <0)// this was set by dash_only
    cmd.cmd_body.set_dash(100);
  // have to turn or dash to targetpos
  else
    get_cmd_to_go2dir_real_data(cmd, targetdir);//TG17
  return true;
}



bool Selfpass2::is_selfpass_safe(int &advantage, const ANGLE targetdir){

  Vector ipos,op;
  double speed;
  int steps, op_num;

  return is_selfpass_safe(advantage, targetdir, speed,ipos, steps, op, op_num);
}

bool Selfpass2::is_selfpass_still_safe(int &advantage, const ANGLE targetdir, double & kickspeed, int &op_num){

  updateSelfpassRiskLevel();

  // used to recheck previous intention. checks nokickc only!
  Vector ipos,op;
  int steps;

  return is_selfpass_safe(advantage, targetdir, kickspeed,ipos, steps, op, op_num, true);
}

bool Selfpass2::is_turn2dir_safe(int &advantage, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const bool check_nokick_only, const int max_dashes){
  return is_turn2dir_safe(advantage,targetdir, kickspeed, targetpos, actual_steps, attacking_op, op_number,
			  WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::me->stamina,
			  WSinfo::ball->pos, WSinfo::ball->vel,
			  check_nokick_only, max_dashes);
}

bool Selfpass2::is_turn2dir_safe(int &advantage, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				 const bool check_nokick_only, const int max_dashes){

  updateSelfpassRiskLevel();

  //  int max_steps = 15;  // default
  int max_steps = MAX_STEPS;  // default

  max_steps = 5;  // rough estimation: 1 kick, 2 turns, then dash

  bool with_kick = false;
  bool target_is_ball = true;
  bool stop_if_turned = true;
  kickspeed = 0.0;

  if((targetdir.diff(myang) <=BODYDIR_TOLERANCE/180.*PI)){
    MRLOG_MOV(0,"turn2dir safe? Hey, I already turned in that direction! targetdir "<<RAD2DEG(targetdir.get_value()));
    max_steps = 2; // try a dash
    stop_if_turned = 0; // don't stop if direction ok -> this is already true
  }

  
  //first, check, without kick movement
  static const float minDistToBall = 1.25*(ServerOptions::player_size + ServerOptions::ball_size);
  simulate_my_movement(targetdir,max_steps,simulation_table,mypos, myvel, myang, mystamina, ballpos, ballvel,
		       with_kick, stop_if_turned); 
  bool tooCloseToBall = simulation_table[1].my_pos.distance(simulation_table[1].ball_pos)<minDistToBall;
  if(!tooCloseToBall){
    simulate_ops_movement(mypos, targetdir, simulation_table, target_is_ball);
    print_table(simulation_table);
    if(check_nokick_selfpass(advantage, simulation_table, targetpos, actual_steps, attacking_op, op_number,ballpos) == true){
      if((targetdir.diff(simulation_table[actual_steps].my_bodydir) <=BODYDIR_TOLERANCE/180.*PI)){
	DBLOG_MOV(0,"Hey, I can turn2dir safely without kick. targetdir "<<RAD2DEG(targetdir.get_value()));
	return true;
      }
    }
  } // not too Close to Ball

  // now check with kick
  with_kick = true;
  target_is_ball = false;
  simulate_my_movement(targetdir,max_steps,simulation_table, mypos, myvel, myang, mystamina, ballpos, ballvel,
		       with_kick, stop_if_turned); 
  simulate_ops_movement(mypos, targetdir, simulation_table, target_is_ball);

  print_table(simulation_table);


  double max_dist = 1000.0;  // do not restrict max dist here.
  if(  determine_kick(advantage, max_dist, simulation_table, targetdir, targetpos, kickspeed, actual_steps, attacking_op, op_number,
		      mypos, myvel, myang, mystamina, ballpos, ballvel) == true){
    if(targetdir.diff(simulation_table[actual_steps].my_bodydir) <=BODYDIR_TOLERANCE/180.*PI){ 
      // at final position, I do look into target direction 
      MRLOG_MOV(0,"Turn 2 Dir Is Safe: Hey, I found a correct kick command to dir "<<RAD2DEG(targetdir.get_value())
		<<" mypos "<<mypos<<" targetpos "<<targetpos<<" kickspeed "<<kickspeed);
      return true;
    }
    else{ // turn not possible
      MRLOG_MOV(0,"Turn 2 dir: I found a correct kick command, but turning not possible in time");
      return false;
    }
  }
  else{ // no kick found
    MRLOG_MOV(0,"Turn2dir Is Safe: Couldn't find an appropriate one step kick to dir "<<RAD2DEG(targetdir.get_value()));
    return false;
  }
  return false;
}


bool Selfpass2::is_selfpass_safe(int &advantage, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const bool check_nokick_only, const int reduce_dashes){
  return is_selfpass_safe(advantage, targetdir, kickspeed, targetpos, actual_steps, attacking_op, op_number,
			  WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::me->stamina,
			  WSinfo::ball->pos, WSinfo::ball->vel, check_nokick_only, reduce_dashes);

}



bool Selfpass2::is_selfpass_safe_without_kick(int &advantage, 
					     const ANGLE targetdir, 
					     double &kickspeed,
					     Vector &targetpos, int &actual_steps,
					     Vector &attacking_op, int & op_number, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				 const int reduce_dashes){

  //  int max_steps = 15;  // default
  int max_steps = MAX_STEPS;  // default

  bool with_kick = false;
  bool target_is_ball = true;
  kickspeed = 0.0;

  static const float minDistToBall = 1.25*(ServerOptions::player_size + ServerOptions::ball_size);

  //first, check, without kick movement
  simulate_my_movement(targetdir,max_steps,simulation_table,mypos, myvel, myang, mystamina, 
		       ballpos, ballvel, with_kick); 
	
  bool tooCloseToBall = simulation_table[1].my_pos.distance(simulation_table[1].ball_pos)<minDistToBall;

  if(!tooCloseToBall){
    simulate_ops_movement(mypos, targetdir, simulation_table, target_is_ball);
    print_table(simulation_table);
    
    if(check_nokick_selfpass(advantage, simulation_table, targetpos, actual_steps,  attacking_op, op_number, ballpos) == true){
      if((targetdir.diff(simulation_table[actual_steps].my_bodydir) 
	  <=BODYDIR_TOLERANCE/180.*PI)){
	return true;
      } // after all, I am looking in targetdir
    } // check nokick selfpass is true
  } // not too close too ball
  MRLOG_MOV(0,"Is Safe without kick: not possible without kick in dir "<<RAD2DEG(targetdir.get_value()));
  return false;
}



bool Selfpass2::is_selfpass_safe_with_kick(int &advantage, const double max_dist,
					  const ANGLE targetdir, 
					  double &kickspeed,
					  Vector &targetpos, int &actual_steps,
					  Vector &attacking_op, int & op_number, 
					  const Vector mypos, const Vector myvel, const ANGLE myang,
					  const double mystamina,
					  const Vector ballpos, const Vector ballvel,
					  const int reduce_dashes){
  
  int max_steps = MAX_STEPS;  // default

  // now check with kick
  bool with_kick = true;
  bool target_is_ball = false;

  simulate_my_movement(targetdir,max_steps,simulation_table,mypos, myvel, myang, mystamina, ballpos, ballvel, with_kick); 
  simulate_ops_movement(mypos, targetdir, simulation_table, target_is_ball);
  
  print_table(simulation_table);


  if(  determine_kick(advantage, max_dist, simulation_table, targetdir, targetpos, kickspeed, actual_steps, attacking_op, op_number, 
		      mypos, myvel, myang, mystamina, ballpos, ballvel,reduce_dashes) == true){
    if(targetdir.diff(simulation_table[actual_steps].my_bodydir) <=BODYDIR_TOLERANCE/180.*PI){ 
      // at final position, I do look into target direction 
      MRLOG_MOV(0,"Is Safe: Hey, I found a correct kick command to dir "<<RAD2DEG(targetdir.get_value())
		<<" targetpos "<<targetpos<<" kickspeed "<<kickspeed);
      return true;
    }
    else{ // turn not possible
      MRLOG_MOV(0,"I found a correct kick command, but turning not possible in time");
      return false;
    }
  }
  else{ // no kick found
    MRLOG_MOV(0,"Is Safe with kick? Couldn't find an appropriate one step kick to dir "<<RAD2DEG(targetdir.get_value())
	      <<" desired kick speed "<<kickspeed);
    return false;
  }
  return false;
}



bool Selfpass2::is_dashonly_safe(int &advantage, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				 const int max_dashes){

  Cmd tmp_cmd;
  
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  double next_bodydir;
  int next_stamina;

  if(targetdir.diff(myang)>5./180.*PI){ // have to turn
    MRLOG_MOV(0,"Targetdir and body dir differ too much; I cannot dash only");
    return false;
  }


  tmp_cmd.cmd_body.set_dash(100);
  Tools::model_cmd_main(mypos,myvel,myang.get_value(),(int)mystamina, ballpos, ballvel,
			tmp_cmd.cmd_body, 
			next_pos,next_vel,next_bodydir,next_stamina, 
			next_ballpos, next_ballvel);



  if(next_pos.distance(next_ballpos)>WSinfo::me->kick_radius){
    MRLOG_MOV(0,"selfpass dash only: I cannot dash AND keep ball in kickrange");
    return false;
  }

  targetpos = next_pos;
  kickspeed = -1;  //indicate for get_cmd that you want to dash
  advantage = 2;
  actual_steps = 1;

  PlayerSet opset;
  opset= WSinfo::valid_opponents;
  opset.keep_and_sort_closest_players_to_point(1, next_ballpos);
  if ( opset.num ==1 ){
    bool result = Tools::is_ball_safe_and_kickable(next_pos,opset[0],
					    next_ballpos, true);
    MRLOG_MOV(0,"selfpass dash only: Is ball safe and kickable says: "<<result);
    attacking_op = opset[0]->pos;
    op_number = opset[0]->number;

    return result;
  }

  return true;
}




bool Selfpass2::is_selfpass_safe(int &advantage, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				 const bool check_nokick_only, const int reduce_dashes){

  updateSelfpassRiskLevel();

  // first, check nokick and if I can really advance with no kick at all.

  if(is_selfpass_safe_without_kick(advantage, targetdir,kickspeed, targetpos, actual_steps, attacking_op, op_number, mypos, myvel, myang, mystamina, ballpos, ballvel, reduce_dashes) == true){
    if (mypos.distance(targetpos) > 3.0){
      // I can advance very nicely
      MRLOG_MOV(0,"Hey, I can considerably move without kick in dir "<<RAD2DEG(targetdir.get_value()));
      return true;
    }
    else if(check_nokick_only == true){
      MRLOG_MOV(0,"Hey, I only test withoutkick, and at least can turn 2 dir "<<RAD2DEG(targetdir.get_value()));
      return true;
    }
  }
  // from here, check no kick failed.
  if(check_nokick_only == true){
    MRLOG_MOV(0,"Hey, Check with nokick only not successful, return false");
    // hmm, withoutkick seems not to be a success!!!
    return false;
  }

  double max_dist = 1000.0; // do not restrict yet

  return is_selfpass_safe_with_kick(advantage, max_dist, targetdir,kickspeed, targetpos, actual_steps, attacking_op, op_number, mypos, myvel, myang, mystamina, ballpos, ballvel, reduce_dashes);
  
}


bool Selfpass2::is_selfpass_safe_max_advance(int &advantage, const double max_dist, const ANGLE targetdir, double &kickspeed, Vector &targetpos, int &actual_steps,
				 Vector &attacking_op, int & op_number, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				 const bool check_nokick_only, const int reduce_dashes){

  updateSelfpassRiskLevel();

  // first, check nokick and if I can really advance with no kick at all.

  int advantage_wk;
  double kickspeed_wk;
  Vector targetpos_wk;
  int actual_steps_wk;
  int op_number_wk;
  Vector attacking_op_wk;
  targetpos = mypos;  // default
  targetpos_wk = mypos;  // default

  bool is_safe_wk = false;

  bool is_safe_nk = is_selfpass_safe_without_kick(advantage, targetdir,kickspeed, targetpos, actual_steps, attacking_op, op_number, mypos, myvel, myang, mystamina, ballpos, ballvel, reduce_dashes);
  
  if(check_nokick_only == false){
    is_safe_wk = is_selfpass_safe_with_kick(advantage_wk, max_dist, targetdir,kickspeed_wk, targetpos_wk, actual_steps_wk, attacking_op_wk, op_number_wk, mypos, myvel, myang, mystamina, ballpos, ballvel, reduce_dashes);  }
  else{
    is_safe_wk = false;
  }

#define MIN_ADVANCE 1.0 // important; otherwise no prorgress is made; agent gets stuck

  if(targetpos.distance(mypos)<MIN_ADVANCE)
    is_safe_nk = false;

  if(targetpos_wk.distance(mypos)<MIN_ADVANCE)
    is_safe_wk = false;


  if(is_safe_wk == false){ // with kick is false; so no kick wins.
    MRLOG_MOV(0,"Hey, Check with with kick not successful, return no kick result:"<<is_safe_nk);
    return is_safe_nk;
  }
  // with kick is safe. now check if it wins for copying
  if(targetpos.distance(mypos) > targetpos_wk.distance(mypos) -0.5 ){ 
    // without kick I go further than with kick
    MRLOG_MOV(0,"Hey, Check with with kick successful; dist: "<<
	      targetpos_wk.distance(mypos)<<
	      " , but no kick gets further:"<<targetpos.distance(mypos));
    return is_safe_nk;
  }
  // wk is safe, and goes further than nokick -> switch to with kick:
  advantage = advantage_wk;
  targetpos = targetpos_wk;
  actual_steps = actual_steps_wk;
  attacking_op = attacking_op_wk;
  op_number = op_number_wk;
  kickspeed = kickspeed_wk;

  MRLOG_MOV(0,"Hey, Check with with kick successful AND WINS! dist: "<<
	    targetpos_wk.distance(mypos)<<
	    " No kick distance:"<<targetpos.distance(mypos));
  return is_safe_wk;
  
}





void Selfpass2::get_cmd_to_go2pos(Cmd &tmp_cmd,const Vector targetpos,const Vector pos, const Vector vel, 
				  const ANGLE bodydir, const int stamina,
				  const PPlayer player){

  if(pos.distance(targetpos) < 2*ServerOptions::player_speed_max){
    // potentially could reach the targetpos by simply dashing; so try this first!
    Vector next_pos, next_vel;
    ANGLE next_bodydir;
    int next_stamina;

    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();
    basic_cmd->set_dash(100);
    basic_cmd->get_cmd(tmp_cmd);	
    Tools::simulate_player(pos,vel,bodydir,stamina,
			   tmp_cmd.cmd_body, 
			   next_pos,next_vel,next_bodydir,next_stamina,
			   player->stamina_inc_max, player->inertia_moment,
			   player->dash_power_rate,
			   player->effort, player->decay);
    if(at_position(next_pos, next_bodydir, player->kick_radius, targetpos)){
      // hey, a dash will do this!!
      return;
    }
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();
  }

  // normal computation
  ANGLE targetdir;
  targetdir = (targetpos - pos).ARG();
  get_cmd_to_go2dir(tmp_cmd, targetdir, pos,vel,bodydir, stamina, player->inertia_moment, 
		    player->stamina_inc_max);
}


void Selfpass2::get_cmd_to_go2dir(Cmd &tmp_cmd,const ANGLE targetdir,const Vector pos, const Vector vel, 
                  const ANGLE bodydir, const int stamina,
                  const double inertia_moment, const double stamina_inc_max){
  // compute right command to dash in targetdir
  //  double moment;

  if(targetdir.diff(bodydir) >BODYDIR_TOLERANCE/180.*PI){ // have to turn
    double moment = (targetdir-bodydir).get_value_mPI_pPI() *
      (1.0 + (inertia_moment * (vel.norm())));
    if (moment > 3.14) moment = 3.14;
    if (moment < -3.14) moment = -3.14;
    basic_cmd->set_turn(moment);
    basic_cmd->get_cmd(tmp_cmd);
    return;
  }
  
  // turned, now dashing
  int dash_power = 100;
  if(stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
    dash_power = (int)stamina_inc_max;
  }
  basic_cmd->set_dash(dash_power);
  basic_cmd->get_cmd(tmp_cmd);
  return;
}

void Selfpass2::get_cmd_to_go2dir_real_data(Cmd &tmp_cmd,const ANGLE targetdir) //TG17
{
  Vector vel = WSinfo::me->vel;
  ANGLE bodydir = WSinfo::me->ang;
  int stamina = WSinfo::me->stamina;
  double inertia_moment = WSinfo::me->inertia_moment;
  double stamina_inc_max = WSinfo::me->stamina_inc_max;

  Vector ballInT3Steps =   WSinfo::ball->pos
                         + WSinfo::ball->vel
                         + ServerOptions::ball_decay * WSinfo::ball->vel
                         + ServerOptions::ball_decay * ServerOptions::ball_decay * WSinfo::ball->vel;
  Vector meAfterDashing( WSinfo::me->ang );
  meAfterDashing += WSinfo::me->pos;
  double deviation
    = Tools::get_dist2_line( WSinfo::me->pos, meAfterDashing, ballInT3Steps );
  double usedTolerance = BODYDIR_TOLERANCE;
  if (deviation < 0.5) usedTolerance *= 2.0;
  if(targetdir.diff(bodydir) > usedTolerance/180.*PI){ // have to turn
    double moment = (targetdir-bodydir).get_value_mPI_pPI() *
      (1.0 + (inertia_moment * (vel.norm())));
    if (moment > 3.14) moment = 3.14;
    if (moment < -3.14) moment = -3.14;
    basic_cmd->set_turn(moment);
    basic_cmd->get_cmd(tmp_cmd);
    return;
  }

  // turned, now dashing
  int dash_power = 100;
  if(stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 100.){
    dash_power = (int)stamina_inc_max;
  }
  basic_cmd->set_dash(dash_power);
  basic_cmd->get_cmd(tmp_cmd);
  return;
}

void Selfpass2::reset_simulation_table(Simtable *simulation_table){
  for(int i=0; i<MAX_STEPS; i++){
    simulation_table[i].valid_at = -1; // invalidate
  }
}


void Selfpass2::simulate_my_movement(const ANGLE targetdir, const int max_steps, Simtable *simulation_table, 
				 const Vector mypos, const Vector myvel, const ANGLE myang,
				 const double mystamina,
				 const Vector ballpos, const Vector ballvel,
				     const bool with_kick, const bool turn2dir_only)
{
  // creates a table with possible positions, when first cmd is a kick
  Cmd tmp_cmd;

  Vector tmp_pos = mypos;
  Vector tmp_vel = myvel;
  Vector tmp_ballvel = ballvel;  
  Vector tmp_ballpos = ballpos;  
  Vector next_pos, next_vel, next_ballpos, next_ballvel;
  ANGLE tmp_bodydir = myang;
  Angle next_bodydir;
  int tmp_stamina = (int) mystamina;
  int next_stamina;

  reset_simulation_table(simulation_table);

  for(int i=0; i<=max_steps && i<MAX_STEPS; i++)
  {
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();
    if(i==0 && with_kick == true)
    {
      // do nothing here; simulate a kick command, which is currently not known.
      basic_cmd->set_turn(0);
      basic_cmd->get_cmd(tmp_cmd); // copy it to tmp_cmd;
    }
    else 
      get_cmd_to_go2dir(tmp_cmd, targetdir,tmp_pos,tmp_vel,tmp_bodydir, tmp_stamina, WSinfo::me->inertia_moment, WSinfo::me->stamina_inc_max);

    // write to table:
    simulation_table[i].valid_at = WSinfo::ws->time; // currently valid
    simulation_table[i].my_pos = tmp_pos;
    simulation_table[i].my_vel = tmp_vel;
    simulation_table[i].my_bodydir = tmp_bodydir;
    simulation_table[i].ball_pos = tmp_ballpos;
    simulation_table[i].ball_vel = tmp_ballvel;
    simulation_table[i].cmd = tmp_cmd;
    if(tmp_pos.distance(tmp_ballpos) <= WSinfo::me->kick_radius - SAFETY_MARGIN)
      simulation_table[i].I_have_ball = true;
    else
      simulation_table[i].I_have_ball = false;
    if(turn2dir_only == true)
    {
      if((targetdir.diff(simulation_table[i].my_bodydir) <=BODYDIR_TOLERANCE/180.*PI))
      {
        // only do steps, until direction is ok.
        return;
      }
    }

    Tools::model_cmd_main(tmp_pos,tmp_vel,tmp_bodydir.get_value(),tmp_stamina, tmp_ballpos, tmp_ballvel,
			  tmp_cmd.cmd_body, 
			  next_pos,next_vel,next_bodydir,next_stamina, next_ballpos, next_ballvel);
      
    tmp_pos = next_pos;
    tmp_vel = next_vel;
    tmp_ballpos = next_ballpos;
    tmp_ballvel = next_ballvel;
    tmp_bodydir = ANGLE(next_bodydir);
    tmp_stamina = next_stamina;
  };  // while
}

bool Selfpass2::at_position(const Vector playerpos, const ANGLE bodydir, const double kick_radius, const Vector targetpos){
   if(playerpos.distance(targetpos) <=  kick_radius) // at position!
     return true;
   Vector translation, new_center;
   translation.init_polar(0.5, bodydir);
   new_center = playerpos + translation;
   if(new_center.distance(targetpos) <= kick_radius) // at position!
     return true;
   return false;
}


int Selfpass2::get_min_cycles2_pos(const Vector mypos, 
                                   const ANGLE targetdir, 
                                   const Vector targetpos, 
                                   const PPlayer player, 
                                   const int max_steps, 
                                   Vector &resulting_pos,
                                   bool forceCalculation)
{
  int minInact = 0, maxInact;
  if (player != NULL)
  {
    WSinfo::get_player_inactivity_interval( player, minInact, maxInact );
  }


  Cmd tmp_cmd;
  Vector tmp_pos = player->pos;
  Vector tmp_vel = player->vel;
  Vector next_pos, next_vel;
  ANGLE tmp_bodydir = player->ang;
  ANGLE next_bodydir;
  int tmp_stamina = (int) player->stamina;
  int next_stamina;

  // ridi: make it more safe:
#if 0 //TG09: war ehedem: 1
  if (player->age_ang >0){
    tmp_bodydir = (targetpos - tmp_pos).ARG();
  }
#endif
//#if 1  // ridi 2006: try this or not, probably too cautious
#if 0 //TG09
  if (player->age_vel >0){
    Vector vel;
    vel.init_polar(ServerOptions::player_speed_max * player->decay, (targetpos - tmp_pos).ARG());
    tmp_vel = vel;
  }
#endif

  resulting_pos = tmp_pos;

  int i;
  if(player->number == 0){
    DBLOG_MOV(0,"checking op player "<<player->number<<" kick radius "<<player->kick_radius
	      <<" inertia_moment "<<player->inertia_moment<<" power rate "<<player->dash_power_rate
	      <<" effort "<<player->effort<<" decay "<<player->decay);
  }

//MRLOG_MOV(0,"SP_COMPARE: "<<max_steps<<" ... "<<tmp_pos.distance(targetpos)<<" vs. "
//  <<((max_steps+2) * 1.25 * ServerOptions::player_speed_max + player->kick_radius));
  
  if (   forceCalculation == false //TG09
      &&   tmp_pos.distance(targetpos) 
         > (max_steps+2) * 1.25 * ServerOptions::player_speed_max + player->kick_radius
      )
  {
    // no chance to get to position in time
    //    MRLOG_DRAW(0,C2D(tmp_pos.x,tmp_pos.y,1.3,"black"));          
    return -1;
  }
  
  if (   forceCalculation == false //TG09
      && targetdir.diff((player->pos-mypos).ARG()) > 150./180. *PI
      && player->pos.distance(mypos) > 3.0 //TG09  
     )
  {
    // MRLOG_DRAW(0,C2D(player->pos.x, player->pos.y,1.3,"red"));
    return -1;
  }

  for(i=0; i<=max_steps && i<=MAX_STEPS; i++)
  {
    // test only: remove or deactivate
    /*
    char infoString[10];
    sprintf(infoString,"%d",max_steps); // this denotes the trajecetory number
    DBLOG_DRAW(0,STRING2D(tmp_pos.x-0.2,tmp_pos.y-0.4,infoString,"ffff00"));
    DBLOG_DRAW(0,C2D(next_pos.x,next_pos.y,player->kick_radius,"ffff00"));      
    */
    // test only: end

    resulting_pos = tmp_pos;
    double radius = player->kick_radius;
    if(player == WSinfo::his_goalie)
      radius = ServerOptions::catchable_area_l;

    if (at_position(tmp_pos, tmp_bodydir, radius ,targetpos) == true)
    {
      // hey, I am already at position
//      return (int)(((float)(i+1) * 1.2)+0.5);//ZUI ZUI_HIGHZUIZUI ZUI ZUI ZUI ZUI
if (WSinfo::me->pos.getX() > FIELD_BORDER_X-17.0 )
  return i;
      return i+cvSelfpassRiskLevel;
    }
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.unset_cmd();

    get_cmd_to_go2pos(tmp_cmd, targetpos,tmp_pos,tmp_vel,tmp_bodydir, tmp_stamina, player);

    if (   i < minInact
        || (   WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 20.0
            && (minInact+maxInact)/2 > 0 )
       )
      tmp_cmd.cmd_body.set_turn(0.0); //simulate tackle inactivity

    Tools::simulate_player( tmp_pos,
                            tmp_vel,
                            tmp_bodydir, 
                            ServerOptions::stamina_max, //tmp_stamina, <== TG09 !!!!!!!!!
                            tmp_cmd.cmd_body, 
                            next_pos,
                            next_vel,
                            next_bodydir,
                            next_stamina,
                            player->stamina_inc_max, 
                            player->inertia_moment,
                            player->dash_power_rate,
                            player->effort, 
                            player->decay);

//MRLOG_MOV(0,"SP_OPP_SIM["<<player->number<<"]: "<<i<<" ... "
//<<"target="<<targetpos<<" stamina="<<next_stamina<<" next_pos="<<next_pos<<" next_vel="<<next_vel<<" next_bodydir="<<next_bodydir
//<<" remDist="<<next_pos.distance(targetpos));

    tmp_pos = next_pos;
    tmp_vel = next_vel;
    tmp_bodydir = ANGLE(next_bodydir);
    tmp_stamina = next_stamina;
  };  // while
  return -1;
}

void Selfpass2::simulate_ops_movement(const Vector mypos, const ANGLE targetdir, Simtable *simulation_table, const bool target_is_ball){
  // for all steps, for all ops: check time to playerpos
  
  // check opponents;
  PlayerSet pset = WSinfo::valid_opponents; // enough, if I do it once.
  Vector closest_op;
  double closest_dist = 1000;

  for(int t=0; t<MAX_STEPS; t++)
  {
    if(simulation_table[t].valid_at != WSinfo::ws->time)
    {
      // entry not valid
      break;
    }
    Vector targetpos = simulation_table[t].my_pos;
    if(target_is_ball == true)
    {
      targetpos = simulation_table[t].ball_pos;
    }
    int best_op_steps = 1000+t;
    Vector best_op_resulting_pos;
    int closest_op_num = -1;
    int best_idx = -1, closest_op_idx = -1;

    for (int idx = 0; idx < pset.num; idx++) 
    {
      int max_steps = t+2; // check for t+2 steps now (ridi06)
      //      int max_steps = t; // check for t+2 steps now (ridi06)
      Vector resulting_pos;
      int op_steps = get_min_cycles2_pos(mypos, targetdir, targetpos, pset[idx], max_steps, 
                                         resulting_pos);
      if (op_steps < 0 )
        op_steps = 1000+t;

      if (op_steps < best_op_steps) // I found an opponent that gets to position in time
      {
        best_op_steps = op_steps;
        best_op_resulting_pos = resulting_pos;
        best_idx = idx;
        closest_dist = 0;
      }
      if(targetpos.distance(resulting_pos) < closest_dist)
      {
        closest_dist = targetpos.distance(resulting_pos);
        closest_op = pset[idx]->pos;
        closest_op_num = pset[idx]->number;
        closest_op_idx = idx;
      }
    } // for all ops idx

    if(best_idx >=0){ // found an intercepting opponent
      simulation_table[t].op_pos = pset[best_idx]->pos;;
      simulation_table[t].op_steps2pos = best_op_steps;
      simulation_table[t].op_num = pset[best_idx]->number;
    }
    else
    {
      simulation_table[t].op_pos = closest_op;
      simulation_table[t].op_num = closest_op_num;
      //      simulation_table[t].op_steps2pos = -1;
      simulation_table[t].op_steps2pos = 1000+t;
      if (   simulation_table[t].op_steps2pos >= 1000 
          && closest_op_idx >= 0 )
      {
        Vector resulting_pos;
        int forcedCyclesValue
          = get_min_cycles2_pos(mypos, targetdir, targetpos, pset[closest_op_idx], 
                                3*MAX_STEPS, resulting_pos, true );//dont return 1000
        if ( forcedCyclesValue >= 0 )
          simulation_table[t].op_steps2pos = forcedCyclesValue; 
        else //would still be 1000
          simulation_table[t].op_steps2pos 
            = pset[closest_op_idx]->pos.distance(targetpos) / pset[closest_op_idx]->speed_max;
      }
    }
  }// for all t
}


void Selfpass2::print_table(Simtable *simulation_table){
  for(int t=0; t<MAX_STEPS; t++){
    if(simulation_table[t].valid_at != WSinfo::ws->time){
      // entry not valid
      break;
    }
    MRLOG_MOV(1,"time "<<t<<" ballpos: "<<simulation_table[t].ball_pos
              <<" mypos: "<<simulation_table[t].my_pos
	      <<" closest op: "<<simulation_table[t].op_pos
	      <<" number "<<simulation_table[t].op_num
	      <<" gets me in "<<simulation_table[t].op_steps2pos<<" steps"<<" have ball: "<<simulation_table[t].I_have_ball);

    // oppos
    char infoString[10];
    sprintf(infoString,"%d",simulation_table[t].op_steps2pos);
    /*
    if(simulation_table[t].op_steps2pos >0){
      DBLOG_DRAW(0,STRING2D(simulation_table[t].op_pos.x-0.2,simulation_table[t].op_pos.y-0.4,infoString,"ff0000")
		 <<C2D(simulation_table[t].op_pos.x,simulation_table[t].op_pos.y,1.1,"ff0000"));
    }
    */

    // mypos
    /*
    sprintf(infoString,"%d",t);

    DBLOG_DRAW(0,C2D(simulation_table[t].my_pos.x,simulation_table[t].my_pos.y,0.1,"0000ff")
	       <<STRING2D(simulation_table[t].my_pos.x-0.2,simulation_table[t].my_pos.y-0.4,infoString,"0000ff"));
    */

  }

  double turn_angle = 0;
  double dash_power = 0;
  double kick_power = 0;
  double kick_angle = 0;

  switch(simulation_table[0].cmd.cmd_body.get_type()){
  case Cmd_Body::TYPE_DASH:
    simulation_table[0].cmd.cmd_body.get_dash(dash_power);
    MRLOG_MOV(1,"DASH "<<dash_power);
    break;
  case Cmd_Body::TYPE_KICK:
    simulation_table[0].cmd.cmd_body.get_kick(kick_power, kick_angle);
    MRLOG_MOV(1,"KICK power "<<kick_power<<" kick dir "<<RAD2DEG(kick_angle));
  break;
  case Cmd_Body::TYPE_TURN:
    simulation_table[0].cmd.cmd_body.get_turn(turn_angle);
    MRLOG_MOV(1,"TURN "<<turn_angle);
    break;
  }
}


bool Selfpass2::determine_kick(int &advantage, double max_dist, Simtable *simulation_table, const ANGLE targetdir,
			       Vector & targetpos, double & targetspeed, int & steps, Vector & attacking_op,
			       int & attacking_num, 
			       const Vector mypos, const Vector myvel, const ANGLE myang,
			       const double mystamina,
			       const Vector ballpos, const Vector ballvel,
			       const int reduce_dashes ){
  int best_t = -1;
  int best_risky_t = -1;

  double current_dist2border = Tools::min_distance_to_border(mypos);

  for(int t=1; t<MAX_STEPS; t++)  // start with t=1, since I know already that I have the ball at t= 0 (NOW!)
  {
    if(simulation_table[t].valid_at != WSinfo::ws->time)
    {
      // entry not valid
      break;
    }
    if(simulation_table[t].my_pos.distance(mypos) >max_dist)
    {
      // go maximum distance only
      break;
    }
    
    //    if(simulation_table[t].op_steps2pos <= t){ // more risky 
    if(simulation_table[t].op_steps2pos <= t)
    { 
      // if opponent gets me at all and he is faster or equally fast at position, stop search.
      // critical point: break or no break:
      // break;  // using break here is the more safe version; probably uses spectacular dribblings
      ;
    }
    else
    { // opponent does not get to position in time, now check if its inside pitch
      if (    (  Tools::min_distance_to_border(simulation_table[t].my_pos) 
               > MIN_DIST_2_BORDER) 
           || (   current_dist2border < MIN_DIST_2_BORDER
               &&   Tools::min_distance_to_border(simulation_table[t].my_pos) 
                  > current_dist2border)
              //TG09
           || (   current_dist2border < MIN_DIST_2_BORDER
               &&   Tools::min_distance_to_border(simulation_table[t].my_pos) 
                  > 0.8*MIN_DIST_2_BORDER
              ) 
         )
      {
        // position is either in pitch, or improves my current situation
        //	if(simulation_table[t].op_steps2pos > t+1) 
        if (simulation_table[t].op_steps2pos > t+1) 
          best_t = t;  // opponent needs more than one step more to reach position
        else
          best_risky_t = t;
      }
    }
  }
  

  if(best_t <0)// didn't find a safe move
    best_t = best_risky_t;

  best_t -= reduce_dashes;  // for a safer life!


  if (best_t <0){ // no position is safe
    steps = -1;
    return false;
  } 

  attacking_op = simulation_table[best_t].op_pos;
  attacking_num = simulation_table[best_t].op_num;
  //  DBLOG_MOV(0,"check with kick selfpass: Attackerpos: "<<attacking_op);

  advantage = simulation_table[best_t].op_steps2pos - best_t;


  // found a safe position
  double summed_decay = 0.0;
  double decay = 1.0;
  for(int t = 0; t<=best_t;t++){ // compute decay; not elegant, but explicit and clear
    summed_decay += decay;
    decay *= ServerOptions::ball_decay;
  };  

  targetpos = simulation_table[best_t].my_pos;
  /*TG09: ZUI: reactivated 3 lines*/
  Vector a_bit_forward;
  a_bit_forward.init_polar(0.8*WSinfo::me->kick_radius/*0.5*/, targetdir);
  targetpos += a_bit_forward;
  

  targetspeed = (ballpos - targetpos).norm()/summed_decay;

  //  DBLOG_MOV(0,"Should kick to "<< targetpos<<" kickspeed "<<targetspeed);
  double speed1, speed2;

  // compute closest opponent to compute (virtual) state for kick command.
  Vector oppos;
  ANGLE opdir;
  int opdirage;
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, ballpos);
  if ( pset.num ){
    oppos = pset[0]->pos;
    opdir = pset[0]->ang;
    opdirage = pset[0]->age_ang;
  }
  else{
    oppos = Vector(1000,1000); // outside pitch
    opdir = ANGLE(0);
    opdirage = 1000;
  }

#if 1
  if (ballpos.distance(WSinfo::ball->pos) > 0.1){
    steps = best_t;
      MRLOG_MOV(0,"check KICK selfpass: SUCCESS: have the ball after "<<best_t<<" cycles! Checking VIRTUAL ballpos");
      return true;
  }
#endif

  onetwostepkick->reset_state(); // reset state
  onetwostepkick->set_state(mypos,myvel,myang,ballpos,ballvel,oppos,opdir,opdirage);
  onetwostepkick->kick_to_pos_with_initial_vel(targetspeed,targetpos);
  //onetwostepkick->kick_in_dir_with_initial_vel(targetspeed,targetdir);
  onetwostepkick->get_vel(speed1,speed2);
  //  MRLOG_DRAW(0,C2D(targetpos.x,targetpos.y,.5,"orange"));          
  if(fabs(speed1-targetspeed)<=0.1){
    // everything's fine!
    onetwostepkick->get_cmd(simulation_table[0].cmd);
    steps = best_t;
    Vector tmp_ballpos =simulation_table[0].ball_pos;
    Vector tmp_ballvel;
    ANGLE kickdir = (targetpos -  tmp_ballpos).ARG();
    tmp_ballvel.init_polar(targetspeed, kickdir);
    
    if(are_intermediate_ballpositions_safe(mypos, targetdir, tmp_ballpos, tmp_ballvel, steps) == true){
      /*TG09*/     print_table(simulation_table);
      MRLOG_MOV(0,"check KICK selfpass: SUCCESS: have the ball after "<<best_t<<" cycles, and  ballpos. safe!");
      return true;
    }
    else{
      MRLOG_MOV(0,"check kick selfpass: NO success: have the ball after "<<best_t<<" cycles, but   INTERMEDIATE steps. NOT safe!");
      return false;
    }
  }
  MRLOG_MOV(0,"Check kick: FALSE. No one-step kick found! ");
  steps = best_t;
  return false;
}


bool Selfpass2::check_nokick_selfpass(int & advantage, Simtable *simulation_table, Vector & targetpos, int & steps, Vector &attacking_op, int & attacking_num, const Vector ballpos){
  int best_t =  -1;
  int best_risky_t =  -1;

  double current_dist2border = Tools::min_distance_to_border(ballpos);

  int max_steps = 0;
  for(int t= MAX_STEPS-1; t>0; t--){  // start with t=1, since I know already that I have the ball at t= 0 (NOW!)
    if(simulation_table[t].I_have_ball== true){
      max_steps = t;
      break;
    }
  }

    DBLOG_MOV(0,"check nokick selfpass: Number of steps I have the ball without kicking: "<<max_steps);
    
  for(int t=1; t<=max_steps; t++){  // start with t=1, since I know already that I have the ball at t= 0 (NOW!)
    if(simulation_table[t].valid_at != WSinfo::ws->time){
      // entry not valid
      break;
    }
    if(simulation_table[t].op_steps2pos <= t)
    { 
      // if opponent gets ball and he is faster or equally fast at position
      DBLOG_MOV(0,"check nokick selfpass: Attackerpos: "<<attacking_op);
      break;
    }
    //    if((simulation_table[t].my_pos - simulation_table[t].ball_pos).norm() <= WSinfo::me->kick_radius - SAFETY_MARGIN){
    if (simulation_table[t].I_have_ball== true)
    {
      // I have the ball at that time.
      if((    Tools::min_distance_to_border(simulation_table[t].ball_pos) > MIN_DIST_2_BORDER) 
           || (   current_dist2border < MIN_DIST_2_BORDER 
               && Tools::min_distance_to_border(simulation_table[t].ball_pos) > current_dist2border))
      {
        // position is either in pitch, or improves my current situation
        if (simulation_table[t].op_steps2pos > t+1)
	      best_t = t;  // opponent needs more than one step more to reach position
        else
          best_risky_t = t;
        DBLOG_MOV(0,"I just set best_t="<<best_t<<" and best_risky_t="<<best_risky_t);
      }  // ball is in pitch
    } // ball is kickable
  } // for all t


  if (best_t <0) // no position is really safe
    best_t = best_risky_t;

  if (best_t <0){ // no position is safe
    steps = best_t;
    return false;
  } 

  attacking_op = simulation_table[best_t].op_pos;
  attacking_num = simulation_table[best_t].op_num;
  DBLOG_MOV(0,"check with nokick: Attackerpos: "<<attacking_op);
  targetpos = simulation_table[best_t].my_pos;
  steps = best_t;
  DBLOG_MOV(0,"check nokick selfpass: SUCCESS: have the ball after "<<best_t<<" cycles, and  no op between!");
  advantage = simulation_table[best_t].op_steps2pos - best_t;
  return true;
}

bool Selfpass2::are_intermediate_ballpositions_safe( const Vector mypos, 
                                                     const ANGLE targetdir,
                                                     Vector tmp_ballpos, 
                                                     Vector tmp_ballvel, 
                                                     const int num_steps)
{
  // for all steps check, if ballposition is safe
  
  // check opponents;
  PlayerSet pset = WSinfo::valid_opponents; // enough, if I do it once.
  Vector resulting_pos;

  for (int t=0; t<=num_steps; t++)
  { 
    DBLOG_DRAW(0,VC2D(tmp_ballpos,.2,"grey"));
    for (int idx = 0; idx < pset.num; idx++) 
    {
      int op_steps = get_min_cycles2_pos( mypos, 
                                          targetdir, 
                                          tmp_ballpos, 
                                          pset[idx], 
                                          t+1, 
                                          resulting_pos);
      //if (   op_steps >=0 ){ // I found an opponent that gets to position in time
      if (   op_steps >= 0
          && op_steps <= t ) 
      { // I found an opponent that gets to position in time
        MRLOG_MOV(0,"check intermediate ballpositions (t="<<t<<"). opponent "<<pset[idx]->number<<" gets ball"
                  <<" in step "<<op_steps);
        MRLOG_DRAW(0,VC2D(tmp_ballpos,0.3,"black"));
        return false;
      }
      else
      {
        //MRLOG_MOV(0,"check intermediate ballpositions (t="<<t<<"). opponent "<<pset[idx]->number<<" gets ball"
        //          <<" in step "<<op_steps);
      }
    } // for all ops idx
    tmp_ballpos += tmp_ballvel;
    tmp_ballvel *= ServerOptions::ball_decay;
  } // for all timesteps
  return true;
}

void  //TG09
Selfpass2::updateSelfpassRiskLevel()
{
  int usedMaxSelfpassRiskLevel = MAXIMAL_SELFPASS_RISKLEVEL;
  if (   WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS
      || WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_GLIDERS)
  {
    usedMaxSelfpassRiskLevel = MAXIMAL_SELFPASS_RISKLEVEL_CAREFUL;
    MRLOG_MOV(0,"Selfpass2: RL: Reduced max risk level of "
      <<usedMaxSelfpassRiskLevel<<" against this opponent.");
  }
  int currentLevelsSuccesses, currentLevelsDontCares, currentLevelsFailures;
  int higherLevelsSuccesses=0, higherLevelsDontCares=0, higherLevelsFailures=100;
  int lowerLevelsSuccesses=100, lowerLevelsDontCares=0, lowerLevelsFailures=0;
  if (  WSmemory::getSelfpassMemoryForRiskLevel( cvSelfpassRiskLevel,
                                                 currentLevelsSuccesses,
                                                 currentLevelsDontCares,
                                                 currentLevelsFailures )  == false )
  {
    MRLOG_MOV(0,"Selfpass2: RL: Error when calling getSelfpassMemoryForRiskLevel.");
    return;
  }
  if ( cvSelfpassRiskLevel < usedMaxSelfpassRiskLevel )
    WSmemory::getSelfpassMemoryForRiskLevel( cvSelfpassRiskLevel + 1,
                                             higherLevelsSuccesses,
                                             higherLevelsDontCares,
                                             higherLevelsFailures );
  if ( cvSelfpassRiskLevel > MINIMAL_SELFPASS_RISKLEVEL )
    WSmemory::getSelfpassMemoryForRiskLevel( cvSelfpassRiskLevel - 1,
                                             lowerLevelsSuccesses,
                                             lowerLevelsDontCares,
                                             lowerLevelsFailures );
  
  //counter variables
  int goodCount  = currentLevelsSuccesses,
      badCount   = currentLevelsFailures +  currentLevelsDontCares,
      totalCount = goodCount + badCount,
      higherGoodCount  = higherLevelsSuccesses,
      higherBadCount   = higherLevelsFailures + higherLevelsDontCares,
      higherTotalCount = higherBadCount + higherGoodCount,
      lowerGoodCount  = lowerLevelsSuccesses,
      lowerBadCount   = lowerLevelsFailures + lowerLevelsDontCares,
      lowerTotalCount = lowerGoodCount + lowerBadCount;
  const int countToBeCertain = 10;

  //ball not in kick range -> no reason for an update
  if ( WSinfo::is_ball_kickable() == false ) return;
  //not enough information for current level
  if ( totalCount < 2 ) return;
  //i recently made a change
  if ( WSinfo::ws->time - cvLastSelfpassRiskLevelChange < 5 ) return;
    
  if ( cvSelfpassRiskLevel < usedMaxSelfpassRiskLevel )
  { 
    //risk level incrementing? => threshold calculation
    double increaseLevelBaseThreshold = 0.75;
    double expectedHigherLevelFailureRate = 0.0;
    if (higherTotalCount > 0)
      expectedHigherLevelFailureRate = (double)higherBadCount / (double)higherTotalCount;
    double higherLevelCertainty = min( 1.0, (double)higherTotalCount / (double)countToBeCertain );
    double increaseLevelAdditionalThreshold
      =   higherLevelCertainty
        * expectedHigherLevelFailureRate
        * (1.0 - increaseLevelBaseThreshold);
    MRLOG_MOV(0,"Selfpass2: RL: pl_"<<WSinfo::me->number<<" curRLev="
      <<cvSelfpassRiskLevel<<" increaseLevelBaseThreshold="<<increaseLevelBaseThreshold
      <<" increaseLevelAdditionalThreshold="<<increaseLevelAdditionalThreshold
      <<" good/total["<<cvSelfpassRiskLevel<<"]="<<((double)goodCount / (double)totalCount));
      
    //risk level incrementing? => do it
    if (        (double)goodCount / (double)totalCount
             >= increaseLevelBaseThreshold + increaseLevelAdditionalThreshold )
    {
      cvSelfpassRiskLevel ++ ;
      cvLastSelfpassRiskLevelChange = WSinfo::ws->time;
      MRLOG_MOV(0,"Selfpass2: RL: pl_"<<WSinfo::me->number
        <<" CHANGE! I [l"<<(cvSelfpassRiskLevel-1)<<":"
        <<currentLevelsSuccesses<<","<<currentLevelsDontCares<<","<<currentLevelsFailures
        <<"] increased the risk level to "<<cvSelfpassRiskLevel
        <<" [l:"<<cvSelfpassRiskLevel<<":"<<higherLevelsSuccesses<<","
        <<higherLevelsDontCares<<","<<higherLevelsFailures<<"] ###> thr="
        <<increaseLevelBaseThreshold + increaseLevelAdditionalThreshold );
      return;
    }
  }  
  
  if ( cvSelfpassRiskLevel > MINIMAL_SELFPASS_RISKLEVEL )
  {
    //risk level decrementing? => threshold calculation
    double decreaseLevelBaseThreshold = 0.6;
    double expectedLowerLevelSuccessRate = 1.0;
    if (lowerTotalCount > 0)
      expectedLowerLevelSuccessRate = (double)lowerGoodCount / (double)lowerTotalCount;
    double lowerLevelCertainty = min( 1.0, (double)lowerTotalCount / (double)countToBeCertain );
    double decreaseLevelAdditionalThreshold
      =   lowerLevelCertainty
        * expectedLowerLevelSuccessRate
        * decreaseLevelBaseThreshold 
        * 0.2;
    MRLOG_MOV(0,"Selfpass2: RL: pl_"<<WSinfo::me->number<<" decreaseLevelBaseThreshold="<<decreaseLevelBaseThreshold
      <<" decreaseLevelAdditionalThreshold="<<decreaseLevelAdditionalThreshold
      <<" good/total["<<cvSelfpassRiskLevel<<"]="<<((double)goodCount / (double)totalCount));
  
    //risk level decrementing? => do it
    if (    (double)goodCount / (double)totalCount
          < decreaseLevelBaseThreshold - decreaseLevelAdditionalThreshold ) 
    {
      cvSelfpassRiskLevel -- ;
      cvLastSelfpassRiskLevelChange = WSinfo::ws->time;
      MRLOG_MOV(0,"Selfpass2: RL: pl_"<<WSinfo::me->number<<" CHANGE! I [l"<<(cvSelfpassRiskLevel+1)
        <<":"<<currentLevelsSuccesses<<","<<currentLevelsDontCares<<","<<currentLevelsFailures
        <<"] decreased the risk level to "<<cvSelfpassRiskLevel
        <<" [l:"<<cvSelfpassRiskLevel<<":"<<lowerLevelsSuccesses<<","
        <<lowerLevelsDontCares<<","<<lowerLevelsFailures<<"] ###> thr="
        <<decreaseLevelBaseThreshold - decreaseLevelAdditionalThreshold );
      return;
    } 
  } 
}

int 
Selfpass2::getRiskLevel()
{
  return cvSelfpassRiskLevel;
}
