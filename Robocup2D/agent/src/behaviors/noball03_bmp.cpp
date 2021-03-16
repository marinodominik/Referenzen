#include "noball03_bmp.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>

#include "../basics/wmstate.h"
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "../policy/planning.h"
//#include "../policy/policy_goalie_kickoff.h"
#include "../policy/policy_tools.h"
#include "ws_memory.h"


#include "../serverparam.h"
 
//#define LOG_INFO2 LOG_POL(4,X)  // log information
//#define LOG_INFO2(X) // do not log information
//#define LOG_INFO3(X) LOG_POL(5,X)  // log information
//#define LOG_INFO3(X) // do not log information
//#define LOG_INFO4 LOG_INFO  // log information
//#define LOG_INFO4(X) // do not log information

//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)

#define LOG_RIDI(YYY,XXX) LOG_POL(YYY,XXX)

//#define DBLOG_POL(YYY,XXX) LOG_POL(YYY,XXX)
#define DBLOG_POL(YYY,XXX) 



#define BASELEVEL 3

#define MAX_STEPS2GO 2 // defines the max. number of steps until the move is terminated and redecided


int oldsteps2go;
bool Noball03::initialized=false;
const float Noball03::GO2BALL_TOLERANCE = 0.5;

/*
Main_Move* Noball03::reconsider_move(Main_Move *current_move){
  IntentionType intention;
  mdpInfo::get_my_intention(intention);

  if (mdpInfo::mdp->play_mode != MDPstate::PM_PlayOn) {
    return current_move;
  }
  last_player_line = get_last_player_line();
  compute_steps2go();
  my_role = DeltaPositioning::get_role(mdpInfo::mdp->me->number);

  if ((intention.type == DEISION_TYPE_BLOCKBALLHOLDER) && 
      (intention.time == mdpInfo::time_current()-1)) {

    Vector old_ang = Vector(intention.p1, intention.p2) - mdpInfo::my_pos_abs();
    //LOG_DEB(0, << "reconsider move --------------");
    int use_long_go2pos = 0;
    Vector new_ang = get_block_position(use_long_go2pos);// - mdpInfo::my_pos_abs();
    //LOG_DEB(0, << _2D << C2D(new_ang.x, new_ang.y, 0.5, "#00ffff"));
    new_ang -= mdpInfo::my_pos_abs();

    //LOG_DEB(0, << "finished ---------------------");
    if (Tools::get_abs_angle(new_ang.arg() - old_ang.arg()) > 40.0) {
      current_move->terminate(); // terminate current move and decide from new
      mdpInfo::clear_my_intention();
      //LOG_DEB(0, << "BS02: Interrupt BLOCK_BALL_HODER - angle changed considerably - decide new");
      return playon();
    }
  } else if ((intention.type == DECISION_TYPE_INTERCEPTBALL) || 
	     (intention.type == DECISION_TYPE_INTERCEPTSLOWBALL)) {
    double suc = Tools::get_tackle_success_probability(mdpInfo::my_pos_abs(),
						      mdpInfo::ball_pos_abs(), 
						      mdpInfo::mdp->me->ang.v);
    if ((suc > 0.5) &&
	(my_role != 0) &&
	mdpInfo::is_object_in_their_penalty_area(mdpInfo::ball_pos_abs())) {
      current_move->terminate(); // terminate current move and decide from new
      mdpInfo::clear_my_intention();
      //LOG_DEB(0, << "BS02: Interrupt INTERCEPT - ball can be tackled - decide new");
      return playon();
    }
  }
  
  //LOG_POLICY_D<<" Magnetic without ball reconsider current move "<<current_move->id_str()
  //      <<" Current Intention Type "<<intention.type;
  
  //LOG_INFO2(" Magnetic without ball reconsider current move "<<current_move->id_str()
  //	      <<" Current Intention Type "<<intention.type);

  return current_move;
}
*/

Noball03::Noball03() {
  /* set default defend params */
 
  // init some params
  time_of_last_go2pos= -10;
  time_of_last_receive_pass = -1;
  time_of_last_kickin = 0;
  stamina_run_back_line = 20;

  //go2ball_tolerance_teammate = 1.0;
  go2ball_tolerance_teammate = 0.0; // ridi: if 1.0, then often conflicts occur
  go2ball_tolerance_opponent = -1.0;
  teamformation_tolerance = 1.0;
  //fine_positioning= 0;
  fine_positioning= 1;  // ridi: activate by default
  //move2ballside= true;
  //DANIEL
  move2ballside = false;
  do_surpress_intercept= true;
  do_tackle=true;

  /* read defend params from config file*/
  ValueParser vp(CommandLineOptions::policy_conf,"PVQ_no_Ball_Policy");
 
  vp.get("teamformation_tolerance",teamformation_tolerance);
  vp.get("fine_positioning",fine_positioning);
  vp.get("fine_positioning",fine_positioning);
  vp.get("stamina_run_back_line", stamina_run_back_line);
  vp.get("move2ballside", move2ballside);
  vp.get("do_surpress_intercept", do_surpress_intercept);
  vp.get("do_tackle", do_tackle);

  noball03_attack = new Noball03_Attack;
  go2pos = new NeuroGo2Pos;
  intercept = new InterceptBall;
  basic_cmd = new BasicCmd;
  face_ball = new FaceBall;
  neurointercept = new NeuroIntercept;
  last_look_to_opponent = -10;
}

Noball03::~Noball03() {
  delete go2pos;
  delete noball03_attack;
  delete intercept;
  delete neurointercept;
  delete basic_cmd;
  delete face_ball;
}

/*
void Noball03::set_aaction_score(AAction &aaction, Vector target, double speed){
  AbstractMDP::set_action(aaction,AACTION_TYPE_SCORE,WSinfo::me->number, 0, target, speed);
}

void Noball03::set_aaction_kicknrush(AAction &aaction, double speed, double dir){
  AbstractMDP::set_action(aaction,AACTION_TYPE_SCORE,WSinfo::me->number, 0, 0, speed, dir);
}

void Noball03::set_aaction_panic_kick(AAction &aaction, Vector target){
  AbstractMDP::set_action(aaction,AACTION_TYPE_PANIC_KICK,WSinfo::me->number, 0, target, 0);
}

void Noball03::set_aaction_backup(AAction &aaction, Vector target){
  AbstractMDP::set_action(aaction,AACTION_TYPE_PANIC_KICK,WSinfo::me->number, 0, target, 0);
}
*/

void Noball03::set_aaction_go2ball(AAction &aaction) {
  AbstractMDP::set_action(aaction, AACTION_TYPE_GO2BALL, WSinfo::me->number, 0, Vector(0), 0);
}

void Noball03::set_aaction_tackle(AAction &aaction, double power) {
  AbstractMDP::set_action(aaction, AACTION_TYPE_TACKLE, WSinfo::me->number, 0, Vector(0), power);
}

void Noball03::set_aaction_go4pass(AAction &aaction, const Vector vballpos, const Vector vballvel) {
  AbstractMDP::set_action_go4pass(aaction,vballpos, vballvel);
}

void Noball03::set_aaction_turn_inertia(AAction &aaction, Angle ang) {
  AbstractMDP::set_action(aaction, AACTION_TYPE_TURN_INERTIA, WSinfo::me->number, 0, Vector(0), 0, ang);
}

void Noball03::set_aaction_face_ball(AAction &aaction) {
  AbstractMDP::set_action(aaction, AACTION_TYPE_FACE_BALL, WSinfo::me->number, 0, Vector(0), 0);
}

void Noball03::set_aaction_face_ball_no_turn_neck(AAction &aaction) {
  AbstractMDP::set_action(aaction, AACTION_TYPE_FACE_BALL, WSinfo::me->number, -1, Vector(0), 0);
}

void Noball03::set_aaction_goto(AAction &aaction, Vector target, double target_tolerance, int consider_obstacles, int use_old_go2pos) {
  /*  if ((my_blackboard.intention.valid_at() == WSinfo::ws->time-1) &&
      (my_blackboard.intention.get_type() == GOTO)) {
    LOG_DAN(0, << "using old go2pos Position!!!");
    AbstractMDP::set_action(aaction, AACTION_TYPE_GOTO, WSinfo::me->number, 0, 
			    my_blackboard.intention.player_target, target_tolerance, 0.0, Vector(0,0), Vector(0,0), consider_obstacles);
  } else {
  */
  LOG_POL(0, << "Set AAction Got2: "<< target);

  my_blackboard.intention.set_goto(target, WSinfo::ws->time);
    if (use_old_go2pos)
      AbstractMDP::set_action(aaction, AACTION_TYPE_GOTO, WSinfo::me->number, -1, 
			      target, target_tolerance, 0.0, Vector(0,0), Vector(0,0), consider_obstacles);
    else 
      AbstractMDP::set_action(aaction, AACTION_TYPE_GOTO, WSinfo::me->number, 0, 
			      target, target_tolerance, 0.0, Vector(0,0), Vector(0,0), consider_obstacles);

    //}
}

#if 0
// meine Version vom 16.6.03
void Noball03::set_aaction_goto(AAction &aaction, Vector target, double target_tolerance, int consider_obstacles) {
  if (my_blackboard.intention.valid_at() == WSinfo::ws->time-1) {
    LOG_DAN(0, << "using old go2pos Position!!!");
    AbstractMDP::set_action(aaction, AACTION_TYPE_GOTO, WSinfo::me->number, 0, 
			    my_blackboard.intention.player_target, target_tolerance, 0.0, Vector(0,0), consider_obstacles);
  } else {
    my_blackboard.intention.set_goto(target, WSinfo::ws->time);
    AbstractMDP::set_action(aaction, AACTION_TYPE_GOTO, WSinfo::me->number, 0, 
			    target, target_tolerance, 0.0, Vector(0,0), consider_obstacles);
  }
}
#endif

void Noball03::reset_intention() {
  my_blackboard.intention.reset();
  LOG_POL(0, << "Noball03 reset intention");
}

void Noball03::check_goto_position(AAction &aaction){
  // ridi 04:  hard coded hack to make a block defense
  Vector target = aaction.target_position;
  Vector ballpos = WSinfo::ball->pos;
LOG_RIDI(0, << "Noball03 check_goto_position go2pos, target (" << target.getX() << "," << target.getY() << ")");

  if(WSmemory::team_last_at_ball() != 0 && WSinfo::is_ball_pos_valid() && WSinfo::ball->pos.getX()<0 && (WSinfo::me->number==2 || WSinfo::me->number==5) && fabs(WSinfo::ball->pos.getY())<fabs(WSinfo::me->pos.getY())&& WSinfo::ball->pos.getX()>-FIELD_BORDER_X+15 &&WSinfo::me->pos.distance(WSinfo::ball->pos)<10){
  int mal=1;
  if(WSinfo::me->number==5)
    mal=-1;
  double min_x=-FIELD_BORDER_X;
  double min_y=FIELD_BORDER_Y*mal;
  double max_x=WSinfo::ball->pos.getX()+10;
  double max_y=WSinfo::ball->pos.getY();
  PlayerSet oppset=WSinfo::valid_opponents;
  PPlayer closest=oppset.closest_player_to_point(WSinfo::ball->pos);
  if(closest!=NULL && closest->pos.distance(WSinfo::ball->pos)<3.5)
    oppset.remove(closest);
  Quadrangle2d check_area = Quadrangle2d(Vector(min_x,min_y), Vector(min_x,max_y),Vector(max_x,max_y),Vector(max_x,min_y));
  oppset.keep_players_in(check_area);
  DBLOG_DRAW(0,check_area);
  if(oppset.num==0){ //jk hack: ball geht ueber innen und keine Gefahr aussen -> go for it
    if(fabs(WSinfo::me->pos.getX()-WSinfo::ball->pos.getX())<3 && (WSinfo::me->pos.distance(WSinfo::ball->pos)>5)){
      aaction.target_position.setY(WSinfo::ball->pos.getY()+(mal*-(0.5*fabs(WSinfo::me->pos.getY()-WSinfo::ball->pos.getY()))));
      }
    else
      aaction.target_position.setY(WSinfo::ball->pos.getY()+WSinfo::ball->vel.getY() );
    LOG_RIDI(0, << "JK HACK: DEFENDER "<<WSinfo::me->number<<" GOES FOR IT(" << target.getX() << "," << target.getY() << ")");
    }
  }
  
  
  // ridi: changed to avoid that behave does not send a cmd
  bool dont_leave_defense_area=true;
  if((WSinfo::me->number==2 || WSinfo::me->number==5) && WSinfo::is_ball_pos_valid() && WSinfo::ball->pos.getX()<-FIELD_BORDER_X+25){
  PlayerSet oppset = WSinfo::valid_opponents;
  PlayerSet opps = WSinfo::valid_opponents;
  PlayerSet mytset = WSinfo::valid_teammates;
    for (int i=6; i<12;i++){
      mytset.remove(WSinfo::get_teammate_by_number(i));
    }
    mytset.remove(WSinfo::get_teammate_by_number(1));
    double min_x=-35;
    double max_x=-FIELD_BORDER_X;
    double min_y=999;
    for (int i=0; i<mytset.num;i++){
      if (mytset[i]->pos.getX()<min_x) min_x=mytset[i]->pos.getX();
      if (mytset[i]->pos.getX()>max_x) max_x=mytset[i]->pos.getX();
      if (WSinfo::me->number!=mytset[i]->number){
        if(fabs(WSinfo::me->pos.getY()-min_y)>fabs(WSinfo::me->pos.getY()-mytset[i]->pos.getY())) min_y=mytset[i]->pos.getY();
      }
    }
    min_x=min_x-5;
    max_x=max_x+3;
    Quadrangle2d check_area = Quadrangle2d(Vector(min_x,WSinfo::me->pos.getY()), Vector(min_x,min_y),Vector(max_x,min_y),Vector(max_x,WSinfo::me->pos.getY()));
    oppset.keep_players_in(check_area);
   // DBLOG_DRAW(0,check_area);
  
  
  if (opps.num!=0){
    PPlayer closest = opps.closest_player_to_point(WSinfo::me->pos);
    if (closest!=0 && closest->pos.distance(WSinfo::me->pos)<2){
    
    //WSpset opps=WSinfo::valid_opponents;
    //opps.keep_players_in_quadrangle(WSinfo::ball->pos,closest->pos,1.);
    Quadrangle2d pass_way = Quadrangle2d(WSinfo::ball->pos,closest->pos,1.0);
    DBLOG_DRAW(0,pass_way);
    oppset.remove(closest);
    if(pass_way.inside(target)&& oppset.num==0 && WSinfo::ball->pos.distance(WSinfo::me->pos)>(WSinfo::ball->pos+WSinfo::ball->vel).distance(WSinfo::me->pos)) {
      LOG_RIDI(0, << "HundJ: move into passway)");
    }
      dont_leave_defense_area=false;
    
    }
  }
  else if (oppset.num==0 && WSinfo::me->pos.distance(WSinfo::ball->pos)>target.distance(WSinfo::ball->pos)&& fabs(target.getY())<15 && WSinfo::me->pos.distance(WSinfo::ball->pos)<6){
    dont_leave_defense_area=false;
    LOG_RIDI(0, << "HundJ: move to ball_owner)");
  }
  
  }
  
  
  if (dont_leave_defense_area){
  if (WSinfo::me->number == 2 && target.getX() < -35){ // corrected position for player 2
    if(target.getY() > 10){
      aaction.target_position.setX( target.getX() );
      aaction.target_position.setY( 10. );
      LOG_RIDI(0, << "Noball03 check_goto_position CORRECTED target (" << aaction.target_position.getX() << ","
	       << aaction.target_position.getY() << ")");
    }
    if(ballpos.getY() <-5 && ballpos.getX() <-42){
      aaction.target_position.setX( Tools::min(-48.,target.getX()) );
      aaction.target_position.setY( Tools::min(6.,target.getY()) );
    }
  }
  if (WSinfo::me->number == 3 && target.getX() < -35){ // corrected position for player 2
    if(target.getY() > 7){
      aaction.target_position.setX( target.getX() );
      aaction.target_position.setY( 7. );
      LOG_RIDI(0, << "Noball03 check_goto_position CORRECTED target (" << aaction.target_position.getX() << ","
	       << aaction.target_position.getY() << ")");
    }
    if(ballpos.getY() <-5 && ballpos.getX() <-42){
      aaction.target_position.setX( Tools::min(-46.,target.getX()) );
      aaction.target_position.setY( Tools::min(3.,target.getY()) );
    }
  }
  if (WSinfo::me->number == 4 && target.getX() < -35){ // corrected position for player 2
    if(target.getY() < -7){
      aaction.target_position.setX( target.getX() );
      aaction.target_position.setY( -7. );
      LOG_RIDI(0, << "Noball03 check_goto_position CORRECTED target (" << aaction.target_position.getX() << ","
	       << aaction.target_position.getY() << ")");
    }
    if(ballpos.getY() >5 && ballpos.getX() <-42){
      aaction.target_position.setX( Tools::min(-46.,target.getX()) );
      aaction.target_position.setY( Tools::max(-3.,target.getY()) );
    }
  }
  if (WSinfo::me->number == 5 && target.getX() < -35){ // corrected position for player 2
    if(target.getY() < -10){
      aaction.target_position.setX( target.getX() );
      aaction.target_position.setY( -10. );
      LOG_RIDI(0, << "Noball03 check_goto_position CORRECTED target (" << aaction.target_position.getX() << ","
	       << aaction.target_position.getY() << ")");
    }
    if(ballpos.getY() >5 && ballpos.getX() <-42){
      aaction.target_position.setX( Tools::min(-48.,target.getX()) );
      aaction.target_position.setY( Tools::max(-6.,target.getY()) );
    }
  }
  }

}


bool Noball03::aaction2cmd(AAction &aaction, Cmd &cmd){
  // ridi 04: make the defense block more robust
  if(aaction.action_type == AACTION_TYPE_GOTO){
    check_goto_position(aaction);
  }

  double speed = aaction.kick_velocity;
  Vector target = aaction.target_position;
  double dir = aaction.kick_dir;
  double target_tolerance = aaction.kick_velocity;
  int consider_obstacles = aaction.advantage;
  int receiver = aaction.target_player;
  
  set_attention(cmd);


  switch(aaction.action_type){
  case AACTION_TYPE_GO2BALL:{
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
    LOG_DAN(0, << "Noball03 aaction2cmd: intercept ball");
    //intercept->get_cmd(cmd);
    //WM04: check if opponent has ball and if so go directly on his pos in order to e.g. tackle
    PPlayer closest=NULL;
    if(WSinfo::valid_opponents.num!=0)
      closest = (WSinfo::valid_opponents).closest_player_to_point(WSinfo::ball->pos);
    if(closest!=NULL && WSinfo::is_ball_kickable_for(closest)){
      go2pos->set_target(closest->pos+closest->vel,0.01);
      go2pos->get_cmd(cmd);
      LOG_RIDI(0, << "Noball03 aaction2cmd: dont intercept but go to ballholder pos directly");
      }
    else
      neurointercept->get_cmd(cmd); // ridi03: use neurointercept
    return true;
    break;
  }
  case AACTION_TYPE_GO4PASS:
#if 0
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
    LOG_DAN(0, << "Noball03 aaction2cmd: intercept ball");
    neurointercept->set_virtual_state(aaction.virtual_ballpos,aaction.virtual_ballvel );
    neurointercept->get_cmd(cmd);
#endif
    noball03_attack->test_go4pass(cmd);
    return true;
    break;
  case AACTION_TYPE_TACKLE:
    LOG_DAN(0, << "Noball03 aaction2cmd: tackle w power " << speed);
    basic_cmd->set_tackle(speed);
    basic_cmd->get_cmd(cmd);
    //cmd.cmd_main.set_tackle(speed);
    return true;
    break;
  case AACTION_TYPE_TURN_INERTIA:
    LOG_DAN(0, << "Noball03 aaction2cmd: turn_inertia to dir " << dir);
    basic_cmd->set_turn_inertia(dir);
    basic_cmd->get_cmd(cmd);
    //turn_inertia->set_turn_angle(dir);
    //turn_inertia->get_cmd(cmd);
    return true;
    break;
  case AACTION_TYPE_FACE_BALL:
    //LOG_POL(0,<<"CallingFaceBall");
    if (receiver == -1) {
      face_ball->turn_to_ball(false);
    }
    else {
      face_ball->turn_to_ball();
    }
    face_ball->get_cmd(cmd);
    //LOG_ERR(0, <<"Noball03 aaction2cmd: face_ball not yet implemented");
    //LOG_DAN(0, <<"Noball03 aaction2cmd: face_ball not yet implemented");
    return true;
    break;
  case AACTION_TYPE_GOTO:
    LOG_RIDI(0, << "OK, lets AACTIPM2 Goto ");
    LOG_DAN(0, << "Noball03 aaction2cmd: go2pos, target (" << target.x << "," << target.y << ")");
    // ridi: changed to avoid that behave does not send a cmd
    if(WSinfo::me->pos.distance(target) > target_tolerance){
      if (receiver == -1)
	go2pos->set_target(target, target_tolerance, consider_obstacles, true);
      else
	go2pos->set_target(target, target_tolerance, consider_obstacles);
      go2pos->get_cmd(cmd);
      return true;
    }
    else{
      face_ball->turn_to_ball(false);
      face_ball->get_cmd(cmd);
      return true;
    }
    break;
  default:
    LOG_DAN(0, <<"Noball03 aaction2cmd: AActionType " << aaction.action_type << " not known");
    LOG_ERR(0, <<"Noball03 aaction2cmd: AActionType " << aaction.action_type << " not known");
    return false;
  }
}



double Noball03::get_last_player_line() {
  PlayerSet pset_tmp= WSinfo::valid_teammates;
  double offside_line;
  pset_tmp.keep_and_sort_players_by_x_from_left(2); //keep oponents in penalty area
  if (pset_tmp.num > 0) {
    if (pset_tmp[0]->number == WSinfo::ws->my_goalie_number)
      offside_line = pset_tmp[pset_tmp.num-1]->pos.getX();
    else 
      offside_line = pset_tmp[0]->pos.getX();
  }
  else
    offside_line= -FIELD_BORDER_X;

  return offside_line;
}


void Noball03::set_attention(Cmd &cmd) {
  if (WSinfo::ball->pos.getX() > -20.0 || last_player_line > -25.0 || WSmemory::team_last_at_ball() != 1) return;

  if (WSinfo::me->number < 2 || WSinfo::me->number > 5)
    return;

  if (cmd.cmd_att.is_cmd_set())
    return;

  int new_att = 6;;

  if (WSinfo::me->number == 2 || WSinfo::me->number == 3) {
    if ( (WSinfo::ws->time % 2 ) == 0 )
      new_att = 8;
    else
      new_att = 6;
  } else if (WSinfo::me->number == 4 || WSinfo::me->number == 5) {
    if ( (WSinfo::ws->time % 2 ) == 0 )
      new_att = 6;
    else
      new_att = 8;
  }

  LOG_DEB(0, << "new attention method ===> " << new_att);

  cmd.cmd_att.set_attentionto(new_att);
}

void Noball03::set_players_to_communicate(Cmd &cmd) {
  if (WSinfo::me->number != 6 && WSinfo::me->number != 8) {
    return;
  }

  LOG_DEB(0, << " ball_pos " << WSinfo::ball->pos << " last_p_line " << last_player_line << " team-last_at_b " << WSmemory::team_last_at_ball());

  if (WSinfo::ball->pos.getX() > -20.0 || last_player_line > -25.0 || WSmemory::team_last_at_ball() != 1) return;

  XYRectangle2d rect;

  if (WSinfo::ball->pos.getX() > -36.0) {
    rect = XYRectangle2d(Vector(-52.5, 8.0), Vector(-52.5+16.5,-8.0));
  } else {
    double upper_y_border = 15.0;
    double lower_y_border = -15.0;
    /*
    if (WSinfo::ball->pos.y > 17.4) {
      upper_y_border = WSinfo::ball->pos.y - 2.4;
      if (upper_y_border > 20.0)
	upper_y_border = 20.0;
    } else if (WSinfo::ball->pos.y < -17.4) {
      lower_y_border = WSinfo::ball->pos.y + 2.4;
      if (lower_y_border < -20.0)
	lower_y_border = -20.0;
	}*/
    rect = XYRectangle2d(Vector(-52.5, upper_y_border), Vector(-52.5+16.5, lower_y_border));
  }

  PlayerSet players4communication;

  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_with_max_age(0);
  pset.keep_players_in(rect);
  players4communication.append(pset);

  PPlayer our_goalie = WSinfo::get_teammate_by_number(mdpInfo::our_goalie_number());

  if ( players4communication.num < 3 ) {
    pset = WSinfo::valid_teammates;
    pset.keep_players_with_max_age( 0 );
    pset.keep_players_in( LEFT_PENALTY_AREA );
    if (our_goalie != NULL)
      pset.remove(our_goalie);
    players4communication.append( pset );
  }
  
  if ( players4communication.num < 3 ) {
    pset = WSinfo::valid_opponents;
    pset.keep_players_with_max_age( 0 );
    players4communication.join( pset );
  }

  if ( players4communication.num < 3 ) {
    pset = WSinfo::valid_teammates_without_me;
    pset.keep_players_with_max_age( 0 );
    players4communication.join( pset );
  }

  LOG_DEB(0, << " number of comm. plaeyers " << players4communication.num);

  cmd.cmd_say.set_players( players4communication );

  for(int i=0; i<players4communication.num;i++){
    LOG_POL(0,<<"My communication set "<<i<<" number "<<players4communication[i]->number<<" pos "<<players4communication[i]->pos);
  }


}

bool Noball03::get_cmd(Cmd &cmd){
  //long ms_time = Tools::get_current_ms_time();
  //local_time = ms_time;


  //basic_cmd->set_turn(0);
  //return basic_cmd->get_cmd(cmd);
  oldsteps2go=9999;

  AAction aaction;

  /* analyse current game state. who will get the ball fastest? */
  compute_steps2go();

  //D DeltaPositioning::update();
  my_role = DeltaPositioning::get_role(WSinfo::me->number);
  last_player_line = get_last_player_line();

  //next_i_pos = Policy_Tools::next_intercept_pos();
  int time;
  Policy_Tools::intercept_min_time_and_pos_hetero(time, next_i_pos,
						  WSinfo::ball->pos, WSinfo::ball->vel,
						  steps2go.opponent_pos, steps2go.opponent_number,
						  false, 1.0, 1.0);


  //LOG_POL(0, << "Time after i_pos computation " << Tools::get_current_ms_time() - ms_time);

#if 0
  /* just for checking correctness: */
  //int our_closest_player = mdpInfo::teammate_closest_to_ball_wme();


  pset_tmp = WSinfo::valid_teammates;
  pplayer_tmp = pset_tmp.closest_player_to_point(WSinfo::ws->ball.pos);

  int our_closest_player = -1;

  if (pplayer_tmp != NULL) {
    our_closest_player = pplayer_tmp.number;  
  }
  
  double ourdist2ball = 1000;

  if(pplayer_tmp != NULL){
    ourdist2ball= pplayer_tmp->pos.distance(WSinfo->ball.pos);
    //ourdist2ball= mdpInfo::teammate_distance_to_ball(our_closest_player);
  }
  if((ourdist2ball <ServerOptions::kickable_area && steps2go.teammate > 0) ||
     (ourdist2ball >ServerOptions::kickable_area && steps2go.teammate == 0)){
    LOG_ERR(0,"Inconsistency: Our dist 2 ball : "<<ourdist2ball
	    <<" steps2go.teammate: "<<steps2go.teammate);
  }

  double theirdist2ball = 1000;//,theirdir;
  //Vector theirpos;

  //mdpInfo::opponent_closest_to_ball(theirdist2ball,theirdir,theirpos);

  pset_tmp = WSinfo::valid_opponents;
  pplayer_tmp = pset_tmp.closest_player_to_point(WSinfo::ws->ball.pos);


  if(pplayer_tmp >= NULL){
    theirdist2ball = pplayer_tmp->pos.distance(WSinfo->ball.pos);
  }

  
  if((theirdist2ball <ServerOptions::kickable_area && steps2go.opponent > 0) ||
     (theirdist2ball >ServerOptions::kickable_area && steps2go.opponent == 0)){
    LOG_ERR(0,"Inconsistency: Their dist 2 ball : "<<theirdist2ball
	    <<" steps2go.opponent: "<<steps2go.opponent);
  }
#endif
  // probably sets turn-neck intention:
  Policy_Tools::check_if_turnneck2ball((int) steps2go.teammate, (int) steps2go.opponent);

  /* updateing game_state */
  //int myteam_steps = (int)Tools::min(steps2go.me, steps2go.teammate);
  //if(mdpInfo::update_game_state((int) myteam_steps, (int) steps2go.opponent)){
  //  LOG_POL(4, <<"PVQ: Game state changed to "<<mdpInfo::get_game_state());  
  //}

  LOG_POL(4, << "PVQ positioning. my pos:: "<< WSinfo::me->pos);
  LOG_POL(3,"PVQNB: stamina state: "<< mdpInfo::get_stamina_state());

  //LOG_POL(0, << "Time before test... computation " << Tools::get_current_ms_time() - ms_time);

  /*
  LOG_DAN(0, << "my attention is set to player : " << WSinfo::ws->my_attentionto);

  if (!WSinfo::is_ball_pos_valid()) {
    LOG_DAN(0, << " ballllll invalid!!!!!");
    }*/

  if (((WSinfo::ws->time - WSinfo::ws->ball.time) >= 7) && 
      //((WSinfo::ball->pos - Vector(-52.5, 0.0)).norm() < 45.0)) {
      (WSinfo::ball->pos.getX() < 0.0) &&
      (WSinfo::me->pos.getX() < 0.0) &&
      (WSinfo::ws->play_mode == PM_PlayOn)) {
    //LOG_DAN(0, << "ball almost invalid, set attention to goalie, nr. " << WSinfo::ws->my_goalie_number);
    int number= WSinfo::ws->my_goalie_number;
    if ( number <= 0 ) //fall back, assume number 1 is the goalie
      number= 1;
    cmd.cmd_att.set_attentionto(number);
  } 

  set_players_to_communicate(cmd);

  if (  1 && (test_ballpos_valid(cmd, aaction) == true) ){
    LOG_POL(1,"Ball pos invalid");
    return aaction2cmd(aaction, cmd);
  }
  else if ( (do_tackle==true) && (test_tackle(aaction) == true) ) {
    LOG_POL(0,"tackle");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_go2ball_value(aaction) == true) ){
    LOG_POL(1,"Go to ball");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_go4pass(aaction) == true) ){
    LOG_POL(1,"Go for pass");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_help_blocking(aaction) == true) ){
    LOG_POL(1,"help blocking");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_help_attack(aaction) == true)){
    LOG_POL(1,"help attack");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_offside(aaction) == true) ){
    LOG_POL(1,"disable offside");
    return aaction2cmd(aaction, cmd);
  }
  else if (  1 && (test_formations(aaction) == true) ){
    LOG_POL(0,"NOBALL Decision: Doing FORMATION");
    return aaction2cmd(aaction, cmd);
  }
  else if ( (test_look2ball(aaction) == true) ) {
    LOG_POL(1,"look to ball");
    return aaction2cmd(aaction, cmd);
  }
  else return false;

  //ms_time = Tools::get_current_ms_time() - ms_time;
  //LOG_POL(0, << "BS02 with Ball needed " << ms_time << "millis");

}

bool Noball03::test_go4pass(AAction &aaction) {
  Cmd cmd;
  Vector ballpos,ballvel;
  if(noball03_attack->test_go4pass(cmd) == false)
    return false;
  LOG_POL(BASELEVEL+0,"PJ: Can get expected pass: Intercepting virtual ball ");
  set_aaction_go4pass(aaction, ballpos,ballvel);
  return true;

#if 0
  Vector ipos,ballpos,ballvel;

  if(Policy_Tools::check_go4pass(ipos,ballpos,ballvel) == false)
    return false;
  LOG_POL(BASELEVEL+0,"PJ: Can get expected pass: Intercepting virtual ball ");
  set_aaction_go4pass(aaction, ballpos,ballvel);
  return true;
#endif
}

bool Noball03::test_help_attack(AAction &aaction) {
  //hilf einen Spieler dem Gegner den Ball abzunehmen
  //LOG_POL(0, << "Time before test_help_attack computation " << Tools::get_current_ms_time() - local_time);

  if(opponent_free_kick_situation() || my_free_kick_situation()){
    return 0;
  }

  //nur sinnvoll, wenn Gegner den Ball im Kickrange hat
  if(!steps2go.ball_kickable_for_opponent) return 0;

  //nur sinnvoll, wenn Gegner den Ball schon laenger haelt (5 Zyklen)
  if(steps2go.opponent_number != last_opponent){
    last_opponent = steps2go.opponent_number;
    time_of_last_chance = WSinfo::ws->time;
  }

  if(WSinfo::ws->time <= time_of_last_chance+5) return 0;

  //bin ich der Zweitschnellste Mittelfelder/Angreifer zum Ball?
  int gonumber = -1;
  for(int i=0;i<22;i++){
    if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM){
      if(DeltaPositioning::get_role(go2ball_list[i].number) >= 1) {
        gonumber=go2ball_list[i].number;
     	if(gonumber != steps2go.teammate_number) break;  
      }
    }
  }
  if(gonumber != WSinfo::me->number) return 0;

  //gehe von hinten an den ballfuehrenden Gegner ran
  //Annahme: mein Teammate geht von vorn ran;
  Vector oppos = steps2go.opponent_pos;
  Vector tepos = steps2go.teammate_pos;
  Vector dipos = oppos - tepos;
  Vector mypos = oppos + dipos;
  if(mypos.getX() <= oppos.getX()) return 0;
  //LOG_DEB(1,"Player: " << gonumber << " helps resolving a deadlock!");
  
  set_aaction_goto(aaction, mypos);
  return true;


}

/** test_ballpos_valid()
    if I don't know where the ball is, first look for it!*/
bool Noball03::test_ballpos_valid(Cmd &cmd, AAction &aaction) {
  //LOG_POL(0, << "Time before test_ballpos computation " << Tools::get_current_ms_time() - local_time);

  if(   WSinfo::ws->play_mode == PM_my_BeforeKickOff
     || WSinfo::ws->play_mode == PM_my_AfterGoal
     || WSinfo::ws->play_mode == PM_my_KickOff
     || WSinfo::ws->play_mode == PM_his_BeforeKickOff
     || WSinfo::ws->play_mode == PM_his_AfterGoal
     || WSinfo::ws->play_mode == PM_his_KickOff){
    return 0;
  }
  if(WSinfo::is_ball_pos_valid())
    return 0;

  if (WSinfo::ball->invalidated == 1 && WSinfo::me->pos.getX() < 0.0 && (WSinfo::ball->pos - WSinfo::me->pos).norm() > 3.0 && WSinfo::ws->play_mode == PM_PlayOn) {
    PPlayer p = WSinfo::get_teammate_by_number(mdpInfo::our_goalie_number());
    if ( p && p->alive ){ //otherwise attentionto causes (wrong command form) messages
      cmd.cmd_att.set_attentionto(mdpInfo::our_goalie_number());
      LOG_DEB(0, << " BALL ALMOST INVALID, SET ATTENTION TO GOALIE");
      //ERROR_OUT << "player number " << WSinfo::me->number << " time " << WSinfo::ws->time 
      //	<< " BALL ALMOST INVALID, SET ATTENTION TO GOALIE";
    }
    return false;
  }

  //LOG_POL(5,"Without ball - WARNING: lost ball pos ");
  mdpInfo::set_my_intention(DECISION_TYPE_SEARCH_BALL);

  LOG_DAN(0, << "Face ball because ball_pos invalid!");
  set_aaction_face_ball(aaction);
  return true;
}

bool  Noball03::test_move2ballside(DashPosition & pos){
  //LOG_POL(0, << "Time before test_move2ballside computation " << Tools::get_current_ms_time() - local_time);
  Vector ballpos = WSinfo::ball->pos;
  double move2ball_dist =6.0;
  double max_move2ball_ypos = ServerOptions::goal_width/2.-1;

  if(move2ballside == false)
    return false;
  if(fabs(pos.getY()) > 10) // my homepos is not close to the middle
    return false;
  if(pos.getX() > -ServerOptions::pitch_length/2. + 15) // my homepos is not in the penalty area
    return false;
  if(ballpos.getX() > -ServerOptions::pitch_length/2. + 30) // ballpos is more than 20m away from goalline
    return false;
  //LOG_DEB(0, << "Correcting defense position from " << pos);
  if(ballpos.getY()<-10){ // ball is on the left wing
    pos.subFromY( move2ball_dist );
    if(pos.getY() < -max_move2ball_ypos)
      pos.setY( -max_move2ball_ypos );
    //LOG_DEB(0, << "to position to ballside" << pos);
    return true;
  }
  if(ballpos.getY()>10){ // ball is on the left wing
    pos.addToY( move2ball_dist );
    if(pos.getY() > max_move2ball_ypos)
      pos.setY( max_move2ball_ypos );
    //LOG_DEB(0, << "to position to ballside" << pos);
    return true;
  }
  return false;
}

DashPosition Noball03::positioning_for_defence_player( const DashPosition & pos ) {
  DashPosition new_pos = pos;
  if(1 && test_cover_attacker(new_pos)) {
    //LOG_DEB(0, << "test_cover_attacker angesprungen!");
  }
  /*
    else if (test_help_attack_as_defender()) {
    //LOG_DEB(0, << "test_help_attack_as_defender angesprungen!");
    }*/
  //else if(test_blocking_ball_holder(new_pos)) ;
  else if(test_move2ballside(new_pos)) {
    //LOG_DEB(0, << "test_move2ballside angesprungen!");
  }
  else if(test_disable_moveup(new_pos)) {
    //LOG_DEB(0, << "test_disable_moveup angesprungen!");
  }
  else if(test_save_stamina(new_pos)) {
    //LOG_DEB(0, << "test_save_stamina angesprungen!");
  }
  else if(test_save_stamina_wm2001(new_pos)) {
    //LOG_DEB(0, << "test_save_stamina_wm2001 angesprungen!"); //the player is required to have enough stamina to reach the 5m line
  }

  //@andi: wenn dash_power=0, dann bewegt Agent sich nicht
  if(WSinfo::me->pos.distance(new_pos) < 0.2 || fabs(new_pos.dash_power) < 0.1){
    new_pos.dash_power = 0.0;
  }
  LOG_DAN(0, << "go to" << "(" << new_pos.x << "," << new_pos.y << ") dash power=" << new_pos.dash_power);
  LOG_DAN(0, << _2D << C2D(new_pos.x,new_pos.y,0.5,"#ffff00"));
  return new_pos;
}


DashPosition Noball03::attack_positioning_for_middlefield_player( const DashPosition & pos ) {
  // ridi 04: attack positioning for midfield player
  // here we enter, if our team was last at ball -> we attack
  Vector mypos = WSinfo::me->pos;   // targetposition; default: my current position
  Vector hp;
  float max_y_tolerance = 15; // could be made player dependend
  float y_variation = 10.; // could be made player dependend
  float min_dist_x = 15.;  // minimum distance to defender

  hp.clone( pos ); // homeposition

  // First, do a correction of the homeposition for attack

  if(WSinfo::me->number == 8){ // right side
    hp.setY( -20.0 ); // default: go 2 the wings
    if(WSinfo::ball->pos.getY() >  5)
      hp.setY( -10.0 ); // if ballpos is on the left wing; go more to the middle
  }
  if(WSinfo::me->number == 6){ // left side
    hp.setY( 20.0 );
    if(WSinfo::ball->pos.getY() < - 5)
      hp.setY( 10.0 ); // if ballpos is on the right wing; go more to the middle
  }

  float width1 = 15;
  Quadrangle2d check_area = Quadrangle2d(mypos,Vector(mypos.getX()-min_dist_x,mypos.getY()) , width1);
  DBLOG_DRAW(0, check_area );
  PlayerSet pset = WSinfo::valid_teammates_without_me;
  pset.keep_players_in(check_area);
  if(pset.num > 0){ // there is a teammate in the area directly behind me -> advance !!!
    PPlayer tcp = pset.closest_player_to_point(hp);
    if(tcp->pos.distance(WSinfo::ball->pos) >10.){ // this player has no ball, so reduce the minimu distance
      min_dist_x = 10.;
    }
    hp.setX( Tools::max(hp.getX(), tcp->pos.getX() + min_dist_x) ); // correct homepos
    if (hp.getX() > WSinfo::his_team_pos_of_offside_line())
      hp.setX( WSinfo::his_team_pos_of_offside_line() -2. ); // stay behind offsideline
  }

  LOG_POL(0, << _2D << VC2D(hp,1.5,"green"));

  if(mypos.getX() <= hp.getX())  // too far behind
    mypos.setX( hp.getX() );
  if(mypos.getY() > hp.getY() + max_y_tolerance) // too far left
    mypos.setY( hp.getY() + max_y_tolerance *.95 );
  if(mypos.getY() < hp.getY() - max_y_tolerance) // too far right
    mypos.setY( hp.getY() - max_y_tolerance *.95 );


  Vector tp = mypos; // default: use corrected target position.

  // now check, if I should find a better position within my region if I could probably get a pass
  if(WSinfo::me->pos.distance(WSinfo::ball->pos) < 35.){
    float max_val = -1;
    int bestpos = 0; // default
    Vector testpos[4];
    testpos[0] = mypos;
    testpos[1] = Vector(hp.getX(),hp.getY());
    testpos[2] = Vector(hp.getX(),hp.getY() + y_variation);
    testpos[3] = Vector(hp.getX(),hp.getY() - y_variation);
    for(int i= 0; i< 4;i++){
      if(testpos[i].getY() >= ServerOptions::pitch_width/2. -1. ) // too far left
	testpos[i].setY( ServerOptions::pitch_width/2. -1. );
      if(testpos[i].getY() <= -(ServerOptions::pitch_width/2. -1.) ) // too far right
	testpos[i].setY( -(ServerOptions::pitch_width/2. -1.) );
      
      double val = Tools::eval_pos (testpos[i]);
      LOG_RIDI(0,"Evaluation pos  "<<i<<" "<<testpos[i]<<"  : "<< val);
      if(val > max_val){
	bestpos = i;
	max_val = val;
      }
    }
    //  LOG_RIDI(0,<< _2D << C2D(tp.x, tp.y ,1.2,"green")); 
    tp = testpos[bestpos];
  }  // end I am a potential candidate for a pass

  LOG_RIDI(0,<< _2D << VC2D(tp ,1.5,"red"));
  LOG_RIDI(0,<< _2D << VC2D(tp ,1.4,"blue"));
  LOG_RIDI(0,"Attack positioning for Midfielder. Best position  "<< tp);

  DashPosition new_pos;
  new_pos.clone( tp );
  new_pos.dash_power = 100.0;


  if(WSinfo::me->pos.distance(new_pos) < 1.0){
    new_pos.dash_power = 0.0;
  }  
  if(WSinfo::ws->play_mode != PM_PlayOn && WSinfo::me->pos.distance(new_pos) < 4.0){
    new_pos.dash_power = 0.0;
  }
  return new_pos;
}

DashPosition Noball03::defense_positioning_for_middlefield_player( const DashPosition & pos ) {
  DashPosition new_pos = pos;
  
  PlayerSet pset_tmp;
  PPlayer p_tmp;
  LOG_RIDI(0,"Fine Positioning Defense for Midfielder");

  pset_tmp = WSinfo::valid_opponents;
  p_tmp = pset_tmp.closest_player_to_point(pos);
  
  int opp = -1;
  if (p_tmp != NULL)
    opp = p_tmp->number;
  
  if(opp != -1){
    Vector lfp = Tools::get_Lotfuss(WSinfo::ball->pos, p_tmp->pos, WSinfo::me->pos);
    if(lfp.distance(pos)<pos.radius && 
       lfp.distance(WSinfo::ball->pos) < p_tmp->pos.distance(WSinfo::ball->pos) &&
       lfp.distance(p_tmp->pos) < p_tmp->pos.distance(WSinfo::ball->pos)) {
      //Zielposition liegt innerhalb des Radiuses
      //        cout << "\nPlayer: " << mdpInfo::mdp->me->number << " verhindert Pass.";
      new_pos.clone( lfp );
    } 
    else{
      //ich bin hinter dem Spieler, also gehe auf herkoemmliche Weise zwischen ihn und den Ball
      Vector a = WSinfo::ball->pos - p_tmp->pos;
      Vector b = p_tmp->pos + 0.1 * a;
      if(b.distance(pos) < pos.radius){
	//          cout << "\nPlayer: " << mdpInfo::mdp->me->number << " verhindert Pass.";
	new_pos.clone( b );
      }
    }
  }

  if(WSinfo::me->pos.distance(new_pos) < 1.0){
    new_pos.dash_power = 0.0;
  }  
  if(WSinfo::ws->play_mode != PM_PlayOn && WSinfo::me->pos.distance(new_pos) < 4.0){
    new_pos.dash_power = 0.0;
  }
  
  
  
  if (WSinfo::is_ball_pos_valid()){
//
PlayerSet opps= WSinfo::valid_opponents;
opps.keep_and_sort_closest_players_to_point(opps.num,WSinfo::ball->pos);
Vector bpos;

bool specialSupermode = false;

if( opps.num != 0 && WSinfo::is_ball_kickable_for( opps[ 0 ] ) )
{
    bpos=WSinfo::ball->pos;
    specialSupermode = true;
}
else if (opps[0]->pos.distance(WSinfo::ball->pos+WSinfo::ball->vel)< ServerOptions::kickable_area){    
  bpos=WSinfo::ball->pos+WSinfo::ball->vel;
   //LOG_RIDI(0,<<"special supermode");
  specialSupermode = true;
}
if( specialSupermode )
{
for(int i=1;i<opps.num;i++){
  Line2d l= Line2d(bpos,opps[i]->pos-bpos);
  if(l.dir.norm()>22)
    continue;
  Vector res;
  Geometry2d::projection_to_line(res,WSinfo::me->pos,l);
  
  if(res.distance(WSinfo::me->pos)<5 && 
     res.distance(opps[i]->pos)<opps[i]->pos.distance(bpos)){
    if(res.distance(bpos)>opps[i]->pos.distance(bpos)){
      Vector dir = l.dir;
      dir.normalize(1.);
      res=opps[i]->pos-dir;
    }
    if(res.distance(WSinfo::me->pos)>5)
      continue;
    LOG_RIDI(0,<<"checking opp: "<<opps[i]->number);
    DBLOG_DRAW(0,VL2D(bpos,
     opps[i]->pos,"aba978"));
    DBLOG_DRAW(0,Circle2d(res,0.4));
    if(opps[i]->pos.getX()<=WSinfo::ball->pos.getX() || (WSinfo::me->pos.getX()<0 && opps[i]->pos.getX()<0)){
      new_pos=DashPosition(res,pos.dash_power,pos.radius);
      return new_pos;
    }
    else
      new_pos=DashPosition(res,pos.dash_power,pos.radius);
    break;
    }
  }
 }
}
 
//int my_stamina = Stamina:: get_state();

    bool keepOldPos = false;

    if( WSinfo::is_ball_pos_valid() && ( WSinfo::ball->pos.getX() < 0 && WSmemory::team_last_at_ball() != 0 ) ) //&& !(stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )){
    {
        switch( WSinfo::me->number )
        {
            case 6 :
            case 8 :
            {
                int mal;
                if( WSinfo::me->number == 8 )
                    mal = 1;
                else
                    mal = -1;
                double my_x_pos;
                double my_y_pos;
                if( ( WSinfo::me->number == 8 && WSinfo::ball->pos.getY() > 0 ) || ( WSinfo::me->number == 6 && WSinfo::ball->pos.getY() < 0 ) )
                {

                    if( WSinfo::ball->pos.getX() > -25 )
                        my_x_pos = WSinfo::ball->pos.getX();
                    else
                        my_x_pos = -25;

                    my_y_pos = mal * 7;
                }
                else
                {
//                    WSpset opps = WSinfo::valid_opponents;
//                    opps.keep_players_in_quadrangle( WSinfo::ball->pos, Vector( WSinfo::ball->pos.x, mal * FIELD_BORDER_Y ), Vector( WSinfo::my_team_pos_of_offside_line(), mal * FIELD_BORDER_Y ), Vector( WSinfo::my_team_pos_of_offside_line(), WSinfo::ball->pos.y ) );
//                    DBLOG_DRAW( 0, Quadrangle2d( WSinfo::ball->pos, Vector( WSinfo::ball->pos.x, mal * FIELD_BORDER_Y ), Vector( WSinfo::my_team_pos_of_offside_line(), mal * FIELD_BORDER_Y ), Vector( WSinfo::my_team_pos_of_offside_line(), WSinfo::ball->pos.y ) ) );
//                    if( opps.num < 1 || opps.closest_player_to_point( Vector( -FIELD_BORDER_X, mal * FIELD_BORDER_Y ) )->pos.distance( WSinfo::ball->pos ) < 2 )
//                        new_pos_set = false;
//                    else
//                    {
//                        PPlayer opp = opps.closest_player_to_point( Vector( -FIELD_BORDER_X, mal * FIELD_BORDER_Y ) );
//
//                        Vector dir = ( opp->pos - WSinfo::ball->pos );
//                        dir.normalize( WSinfo::ball->pos.distance( opp->pos ) * 0.5 );
//                        new_pos.x = ( WSinfo::ball->pos + dir ).x;
//                        new_pos.y = ( WSinfo::ball->pos + dir ).y;
//                    }

                    if( -FIELD_BORDER_X + 10 < WSinfo::my_team_pos_of_offside_line() + 2 )
                        my_x_pos = WSinfo::my_team_pos_of_offside_line() + 2;
                    else
                        my_x_pos = -FIELD_BORDER_X + 10;
                    my_y_pos = -19 * mal;
                }
                //if(new_pos_set){
                new_pos.setXY( my_x_pos, my_y_pos );
                break;
                //}
            }
            case 7 :
            {
                if( fabs( WSinfo::ball->pos.getY() ) < 15 )
                {
                    PlayerSet team = WSinfo::valid_teammates;
                    team.keep_players_in_circle( WSinfo::ball->pos, 5 );
                    for( int i = 0; i < team.num; i++ )
                        if( DeltaPositioning::get_role( team[ i ]->number ) != 0 )
                            keepOldPos = true;
                    if( !keepOldPos )
                    {
                        new_pos.clone( WSinfo::ball->pos );
                    }
                    break;
                }
                else
                {
                    Vector dest7 = Vector( -FIELD_BORDER_X + 18, 0 );
                    if( WSinfo::me->pos.distance( dest7 ) < 3 )
                    {
                        new_pos.clone( WSinfo::me->pos );
                    }
                    else
                    {
                        if( dest7.getX() < WSinfo::my_team_pos_of_offside_line() + 2 )
                            new_pos.setX( WSinfo::my_team_pos_of_offside_line() + 1 );
                        else
                            new_pos.setX( dest7.getX() );
                        new_pos.setY( dest7.getY() );
                    }
                }
            } // case 7 :
        } // switch
    } // surrounding if

if(WSinfo::is_ball_pos_valid() && (WSinfo::ball->pos.getX() >0 && WSmemory::team_last_at_ball() != 0) && WSinfo::ball->pos.getX()<WSinfo::me->pos.getX()){
  new_pos.setX(WSinfo::ball->pos.getX()); //MEGAHACK!!!!!!!!!!!
}

//const _wm_player & me= my_team[WM::my_number];
//Msg_player_type const* player_type= ServerParam::get_player_type( me->tye );
if(WSinfo::me->stamina-WSinfo::me->pos.distance((Vector)new_pos)*WSinfo::me->stamina_demand_per_meter<1500){
LOG_RIDI(0,<<"save stamina: go2 pos: "<<(Vector)pos);
return pos;
}
else{
LOG_RIDI(0,<<" go2 newpos: "<<(Vector)new_pos);
DBLOG_DRAW(0,VL2D(WSinfo::me->pos,
     new_pos,"aba978"));
return new_pos;
}
}

DashPosition Noball03::positioning_for_middlefield_player( const DashPosition & pos ) {
  // ridi 04: do something special for an attacking midfielder
  // if(mdpInfo::is_my_team_attacking()) // this is not properly working!!!
  if(WSmemory::team_last_at_ball() == 0) // we attack
    return attack_positioning_for_middlefield_player(pos ) ;
  else
    return defense_positioning_for_middlefield_player(pos ) ;


  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  // this code is currently NOT USED: ridi 04!
  
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  DashPosition new_pos = pos;
  
  PlayerSet pset_tmp;
  PPlayer p_tmp;

  //wenn der Ball laengere Zeit vor uns ist und wir im Angriff sind, dann bewege ich mich auch nach vorn
  if(mdpInfo::is_my_team_attacking() && WSinfo::ball->pos.getX() > pos.getX()+10){
    double dx = pos.radius / 3.0;
    Vector opponents[NUM_PLAYERS];

    //int opponents_num= mdpInfo::opponents_within_range(pos, pos.distance(mdpInfo::ball_pos_abs()),NUM_PLAYERS,opponents);

    pset_tmp = WSinfo::valid_opponents;
    pset_tmp.keep_players_in_circle(pos, pos.distance(WSinfo::ball->pos));

    int opponents_num = pset_tmp.num;
    for (int i=0; i < pset_tmp.num; i++) {
      opponents[i] = pset_tmp[i]->pos;
    }

    opponents_num= Tools::min(opponents_num,NUM_PLAYERS); //just for sure
    double pos1_val = pos_offence_eval(Vector(pos.getX()+dx,     pos.getY()), WSinfo::ball->pos,opponents_num, opponents);
    double pos2_val = pos_offence_eval(Vector(pos.getX()+1.9*dx, pos.getY()), WSinfo::ball->pos,opponents_num, opponents);
    double pos3_val = pos_offence_eval(Vector(pos.getX()+2.9*dx, pos.getY()), WSinfo::ball->pos,opponents_num, opponents);
    if(pos1_val >= pos2_val && pos1_val >= pos3_val) new_pos.addToX(     dx );
    if(pos2_val >= pos1_val && pos2_val >= pos3_val) new_pos.addToX( 1.9*dx );
    if(pos3_val >= pos1_val && pos3_val >= pos2_val) new_pos.addToX( 2.9*dx );
    //    cout << "\nPlayer: " << mdpInfo::mdp->me->number << " is fine Positioning!";
  }

  //wenn wir in der Verteidigung, versuche mich zwischen Ball und Gegner zu stellen
  if(!mdpInfo::is_my_team_attacking()){
    
    //int opp = mdpInfo::opponent_closest_to(pos);

    pset_tmp = WSinfo::valid_opponents;
    p_tmp = pset_tmp.closest_player_to_point(pos);

    int opp = -1;
    if (p_tmp != NULL)
      opp = p_tmp->number;

    if(opp != -1){
      Vector lfp = Tools::get_Lotfuss(WSinfo::ball->pos, p_tmp->pos, WSinfo::me->pos);
      if(lfp.distance(pos)<pos.radius && 
         lfp.distance(WSinfo::ball->pos) < p_tmp->pos.distance(WSinfo::ball->pos) &&
         lfp.distance(p_tmp->pos) < p_tmp->pos.distance(WSinfo::ball->pos)) {
        //Zielposition liegt innerhalb des Radiuses
	//        cout << "\nPlayer: " << mdpInfo::mdp->me->number << " verhindert Pass.";
        new_pos.clone( lfp );
      } else{
        //ich bin hinter dem Spieler, also gehe auf herkoemmliche Weise zwischen ihn und den Ball
        Vector a = WSinfo::ball->pos - p_tmp->pos;
        Vector b = p_tmp->pos + 0.1 * a;
        if(b.distance(pos) < pos.radius){
	  //          cout << "\nPlayer: " << mdpInfo::mdp->me->number << " verhindert Pass.";
          new_pos.clone( b );
        }
      }
    }
  }

  if(WSinfo::me->pos.distance(new_pos) < 1.0){
    new_pos.dash_power = 0.0;
  }  
  if(WSinfo::ws->play_mode != PM_PlayOn && WSinfo::me->pos.distance(new_pos) < 4.0){
    new_pos.dash_power = 0.0;
  }
  return new_pos;

}

/*****************************************************************************/

double Noball03::pos_offence_eval(const Vector & pos, const Vector ball_pos, int size, Vector * opponents) {
  double eval= 0.0;
  for (int i= 0; i< size; i++) {
    double dum;
    if ( ball_pos.distance(pos) + 2.0 < ball_pos.distance(opponents[i]) ) 
      dum= 0.0;
    else {
      //dum= pos.distance( opponents[i] );
      //dum= exp( -dum*dum/20.0);
      Vector diff= pos - opponents[i];
      Vector ball_2_opponent= opponents[i] - ball_pos;
      diff.rotate(  - ball_2_opponent.arg() );
      dum = exp ( - ( diff.getX()*diff.getX()/10.0 + diff.getY()*diff.getY())/20.0 );
#if 0
      double tmp = pos.distance( opponents[i] );
      if ( tmp > 15.0 ) 
	dum += (tmp -15.0) / tmp;
#endif
    }
    eval += dum;
  }
  return eval;
}

#if LOG_OFFENCE
#include <strstream.h>
#endif
DashPosition Noball03::positioning_for_offence_player( const DashPosition & pos ) {
  const Vector my_pos = WSinfo::me->pos;
  const Vector target_pos = Vector(pos);
  const Vector target_vec = target_pos-my_pos;
  const double  target_dist= target_vec.norm();
  const double radius = pos.radius;
  const Vector ball_pos= WSinfo::ball->pos;
  //Vector ball_vel= mdpInfo::ball_vel_abs();
  const double offence_line = DeltaPositioning::get_my_offence_line();


  DashPosition ret_pos( my_pos, 0.0, pos.radius ); //default: do nothing 
  if ( ret_pos.getX() > offence_line ) //but avoid offside positions
    ret_pos.setX( offence_line );


  if((fabs(my_pos.getX() -target_pos.getX()) <8.) && // I am approximately at my height position
     (WSinfo::me->stamina <= ServerOptions::stamina_max - 500)){
    if((WSmemory::team_last_at_ball() != 0 && (ball_pos-my_pos).norm() >30) || 
       (WSmemory::team_last_at_ball() != 0 && ball_pos.getX() < -10) || // op. act. in our half
       (ball_pos.getX() < -ServerOptions::pitch_length/2. + 20)){  // or the ball is in penaly region
      //LOG_ERR(0,"I'm an attacker and there's no reason to move -> Stay!");
      return ret_pos;
    }
  }
  LOG_POL(4, << "fine pos, target dist: "<<target_dist<<" ball dist: "<<my_pos.distance( ball_pos));

  
  if ( target_dist > radius ) //if not in target_pos circle, run to the center of the circle
    return pos;

  if ( my_pos.distance( ball_pos ) > 35.0 && ball_pos.getX() < 25.0 ) {
    if ( my_pos.distance(pos) < 5.0 )
      return ret_pos;
    return pos;
  }

  //int dum = mdpInfo::teammate_closest_to_me();
  PlayerSet pset_tmp = WSinfo::valid_teammates;
  PPlayer p_tmp = pset_tmp.closest_player_to_point(WSinfo::me->pos);

  //int dum = p_tmp->number;

  Vector closest_teammate = Vector(0.0, 0.0);
  if (p_tmp != NULL) closest_teammate = p_tmp->pos;
  const double min_distance_to_teammates = 7.5;
  Vector opponents[NUM_PLAYERS];

  //int opponents_num= mdpInfo::opponents_within_range(pos, pos.distance(ball_pos),NUM_PLAYERS,opponents);
  //opponents_num= Tools::min(opponents_num,NUM_PLAYERS); //just for sure

  pset_tmp = WSinfo::valid_opponents;
  pset_tmp.keep_players_in_circle(pos, pos.distance(ball_pos));

  int opponents_num = pset_tmp.num;
  for (int i=0; i < pset_tmp.num; i++) {
    opponents[i] = pset_tmp[i]->pos;
  }
  
  int angle_steps= 18;
  int dist_steps= 5;
  Vector best_pos= target_pos;
  double best_val= pos_offence_eval(best_pos, ball_pos, opponents_num, opponents);
  if ( closest_teammate.distance(best_pos) < min_distance_to_teammates )
    best_val= best_val+ 0.5;

#if LOG_OFFENCE
  static char buffer_2d[2048];
  strstream stream_2d(buffer_2d,2048);
#endif
  for (int d= 0; d< dist_steps; d++)
    for (int a= 0; a< angle_steps; a++) {
      Vector dum_pos;
      double dum_val;
      dum_pos.init_polar( double(d + 1)/double(dist_steps) * radius, double(a)/double(angle_steps) * 2.0 * PI);
      dum_pos = target_pos + dum_pos;

      if ( ball_pos.getX() < my_pos.getX() && ball_pos.getX() < 36.0
	   && dum_pos.getX() < target_pos.getX() - 2.0) //falls Ball hinten und ausserhalb des gegnerischen Strafraums, versuche nur vordere Positionen
	continue;

      if ( dum_pos.getX() > offence_line ||  //fall position nicht gueltig
	   !DeltaPositioning::is_position_valid( dum_pos ) ) {
	//LOG_DIRECT( << "!!!!! pos" << dum_pos.x << " " << dum_pos.y << " not valid");
	continue;
      }

      if ( closest_teammate.distance(dum_pos) < min_distance_to_teammates ) //don't crash with teammates
	continue;

      dum_val= pos_offence_eval(dum_pos, ball_pos, opponents_num, opponents);
      if (dum_val < best_val) {
	best_val= dum_val;
	best_pos= dum_pos;
      }
#if LOG_OFFENCE
      stream_2d << P2D(dum_pos.x,dum_pos.y,"black");
#endif
    }
#if LOG_OFFENCE
  LOG_DIRECT( << "OFFENCE FINE POSITIONING ------------------------------------------------");
  LOG_DIRECT( << "best_pos" << best_pos.x << " " << best_pos.y << " best_val= " << best_val);
  stream_2d << C2D( best_pos.x , best_pos.y, 2.0 ,"blue");
#endif  
  if ( my_pos.distance( best_pos ) < 3.0 
       && fabs(best_val - pos_offence_eval(my_pos, ball_pos, opponents_num, opponents)) < 0.2 ) {
#if LOG_OFFENCE
    LOG_DIRECT( << "Keeping old pos, because diff= " << fabs(best_val - pos_offence_eval(my_pos, ball_pos, opponents_num, opponents)) );
    stream_2d << '\0';
    LOG_DIRECT_2D( << buffer_2d);
#endif
    return ret_pos;
  }

  // ridi: 17.6.02: no get free move for offence (should be done in pj)
  return pos;

  ret_pos= DashPosition(best_pos, pos.dash_power, pos.radius);

#if LOG_OFFENCE
  stream_2d << C2D( ret_pos.x , ret_pos.y, 1.8 ,"green");
#endif
  if ( time_of_last_go2pos + 1 >= WSinfo::ws->time )  //last decision was also a go2pos
    if ( pos_of_last_go2pos.getX() <= offence_line
	 &&  DeltaPositioning::is_position_valid( pos_of_last_go2pos )
	 && closest_teammate.distance(pos_of_last_go2pos) >= min_distance_to_teammates  //don't crash with teammates
	 && fabs(best_val - pos_offence_eval(pos_of_last_go2pos, ball_pos, opponents_num, opponents)) < 0.2) {// don't change pos (and especially the dir) if the old pos was also approx. ok
      ret_pos= DashPosition(pos_of_last_go2pos,pos.dash_power, pos.radius);
#if LOG_OFFENCE
      stream_2d << C2D( ret_pos.x , ret_pos.y, 1.6 ,"red");
#endif
    }

#if LOG_OFFENCE
  LOG_DIRECT( << "ret_pos" << ret_pos.x << " " << ret_pos.y << " ret_val= " << pos_offence_eval(ret_pos, ball_pos, opponents_num, opponents));
  stream_2d << '\0';
  LOG_DIRECT_2D( << buffer_2d);
#endif  
  return ret_pos;
}

   
void Noball03::compute_steps2go(){
  /* calculate number of steps to intercept ball, for every player on the pitch, return sorted list */
  Policy_Tools::go2ball_steps_update();
  go2ball_list = Policy_Tools::go2ball_steps_list();

  steps2go.me = 1000;
  steps2go.opponent = 1000;
  steps2go.opponent_number = 0;
  steps2go.teammate = 1000;
  steps2go.teammate_number = 0;
  steps2go.my_goalie = 1000;

  
  /* get predicted steps to intercept ball for 
   *   me, 
   *   my goalie , 
   *   fastest teammate and 
   *   fastest opponent */
  for(int i=0;i<22;i++)
  {
    if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM)
    {
      if(WSinfo::me->number == go2ball_list[i].number)
      {
      	steps2go.me = go2ball_list[i].steps;
      }
      else 
        if (mdpInfo::our_goalie_number() == go2ball_list[i].number)
        {
         	steps2go.my_goalie = go2ball_list[i].steps;	
        }
        else 
          if (steps2go.teammate > go2ball_list[i].steps)
          {
          	steps2go.teammate = go2ball_list[i].steps;
           	steps2go.teammate_number = go2ball_list[i].number;
          }
    }
    if (go2ball_list[i].side == Go2Ball_Steps::THEIR_TEAM)
    {
      if(steps2go.opponent > go2ball_list[i].steps)
      {
      	steps2go.opponent = go2ball_list[i].steps;
       	steps2go.opponent_number = go2ball_list[i].number;
      }
    } 
  }

oldsteps2go=steps2go.me;
  char buffer[200];
  sprintf(buffer,"Go2Ball Analysis: Steps to intercept (Me %.2d) (Teammate %d %.2d) (Opponent %d %.2d)", 
	  steps2go.me, steps2go.teammate_number, steps2go.teammate,  steps2go.opponent_number,  
	  steps2go.opponent);  
  LOG_POL(0, << buffer);

  //LOG_POL(0, << " intercept says I need " << neurointercept->get_steps2intercept() << " steps");

  PPlayer p_tmp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  if (p_tmp != NULL) {
    steps2go.opponent_pos = p_tmp->pos;
    steps2go.ball_kickable_for_opponent = WSinfo::is_ball_kickable_for(p_tmp);
  }
  else {
    steps2go.opponent_pos = Vector(0.0, 0.0);
    steps2go.ball_kickable_for_opponent = false;
  }

  p_tmp = WSinfo::get_teammate_by_number(steps2go.teammate_number);
  if (p_tmp != NULL) {
    steps2go.teammate_pos = p_tmp->pos;
    steps2go.ball_kickable_for_teammate = WSinfo::is_ball_kickable_for(p_tmp);
  }
  else {
    steps2go.teammate_pos = Vector(0.0, 0.0);
    steps2go.ball_kickable_for_teammate = false;
  }

  int mysteps2intercept =  neurointercept->get_steps2intercept();
  if (mysteps2intercept >0 && mysteps2intercept !=steps2go.me && !steps2go.ball_kickable_for_opponent) {
    DBLOG_POL(0,"WARNING! Old intercept estimation "<<steps2go.me  
	      <<" differs from exact estimation "<<mysteps2intercept<<" DO CORRECTION");
    steps2go.me = mysteps2intercept;
  }


  //steps2go.ball_kickable_for_opponent = WSinfo::is_ball_kickable_for(WSinfo::get_opponent_by_number(steps2go.opponent_number));
  //steps2go.ball_kickable_for_teammate = WSinfo::is_ball_kickable_for(WSinfo::get_teammate_by_number(steps2go.teammate_number));

}

bool Noball03::our_team_is_faster(){
  if(steps2go.me < steps2go.opponent+go2ball_tolerance_opponent){
    return true;
  }
  if(steps2go.teammate <= steps2go.opponent+go2ball_tolerance_opponent){
    return true;
  }
  return false;
}

bool Noball03::i_am_fastest_teammate(){
  /*
  if ( (DeltaPositioning::get_role(WSinfo::me->number) == 0) &&
       (DeltaPositioning::get_role(steps2go.teammate_number) == 0) &&
       (steps2go.me >= steps2go.opponent) &&
       (steps2go.teammate >= steps2go.opponent)) {
    double my_dist = (DeltaPositioning::get_position(WSinfo::me->number)-next_i_pos).norm();
    double teammate_dist = (DeltaPositioning::get_position(steps2go.teammate_number)-next_i_pos).norm();
    if (my_dist < teammate_dist) {
      return true;
    } else {
      return false;
    }
    }*/
  if(WSinfo::ball->pos.getX()<-30 && WSinfo::me->number<6 && abs(steps2go.me-oldsteps2go)>10)
    return true;  //jk Hack in order to be safe in front of our goal
  if(steps2go.me < steps2go.teammate+go2ball_tolerance_teammate) // I am the fastest
    return true;
  if(steps2go.me == steps2go.teammate+go2ball_tolerance_teammate) // Me and teammate have same distance
    // if(steps2go.teammate_number>mdpInfo::mdp->me->number)
      return true;
  return false;
}



bool Noball03::attack_fastest_opponent_as_defender() {
  double defense_line_ball_offset = 7.5;
  double ball_distance_to_attack = 7.0;

  defense_line_ball_offset = DeltaPositioning::get_defence_line_ball_offset();

  if (defense_line_ball_offset > 5.5)
    ball_distance_to_attack = defense_line_ball_offset + 3.0;

  Vector ball_pos = WSinfo::ball->pos;
  if (WSinfo::ball->vel.getX() > 0.0) {
    ball_pos = next_i_pos;
  }

  if ( (my_role == 0) &&
       (steps2go.me >= steps2go.opponent) &&
       //(WSinfo::me->pos.getX() > -30.0) &&
       (WSinfo::ball->pos.getX() - ball_distance_to_attack > last_player_line) ) {
    LOG_DAN(0, "don't intercept, I am not the (global) fastest");
    return 0;
  }

  if ( (my_role == 0) &&
       (steps2go.me >= steps2go.opponent) &&
       (last_player_line >= -10.0) &&
       (defense_line_ball_offset > 6.0) &&
       (WSinfo::ball->pos.getX() - 3.0 > last_player_line) ) {
       //(next_i_pos.x > last_player_line) ) {
    return 0;
  }

  return 1;

}




/* make go2ball decision 
   attention: expects go2ball_list and other go2ball 
   decision stuff initialized correctly.

   in general the fastest player is going to intecept the ball. 
   information about the expected steps to intercept the ball can 
   be found in the steps2go data structure for all players on the
   pitch (see compute_steps2go()).

   for several reasons greedy intercepting can be disabled.
   (1) game state forbids going to the ball completely (opponent free kick etc.)
   (2) in free kick situations there is enough time 
       to selet the player to go to the ball by a more 
       sophisticated criterion
   (3) in situations where the opponent is faster to the ball
       intercepting can lead to a poor defending behaivour and
       be very dangerous. in such cases intercepting is disabled 
       and replaced by blocking.
*/
bool Noball03::test_go2ball_value(AAction &aaction){
  //LOG_POL(0, << "Time before test_g02ball computation " << Tools::get_current_ms_time() - local_time);

  //LOG_DEB(0, << _2D << C2D(next_i_pos.x, next_i_pos.y,0.5,"#ff0000"));
  //LOG_DEB(0, << _2D << C2D(next_intercept_pos_NN().x, next_intercept_pos_NN().y, 0.5, "#0000ff"));

  /* (1) Dont go to ball if there is a free kick, or kick in 
     or something like this for the opponent team or a game 
     state where going to the ball makes no sense at all*/
  if(opponent_free_kick_situation()){
    LOG_DEB(0, << "dont go to ball, opponent_free_kick_sit");
    return 0;
  }
  
  /* (2) if there is a free kick, or kick in or something like this 
     for our team a midfielder should kick in (why?) */
  if(my_free_kick_situation() && Policy_Tools::use_clever_moves && !(Policy_Tools::goaliestandards && (WSinfo::ws->play_mode == PM_my_GoalKick || WSinfo::ws->play_mode == PM_my_GoalKick)) && !(WSinfo::ws->play_mode == PM_my_KickIn) && !(WSinfo::ws->play_mode == PM_my_FreeKick)) {
    return 0;
  }

  if(my_free_kick_situation()){
    LOG_DEB(0, << "dont go to ball, my_free_kick_sit");
    return kick_in(aaction, kick_in_by_formation());
  } else{
    time_of_last_kickin = 0;
  }
    
  
  /* if we reached this line we are in play_on playmode. */ 

  if (!i_am_fastest_teammate()) {
    LOG_DEB(0, << "dont go to ball, im not the fastest teammate");
    /* dont go to ball because a teammate is faster!*/
    return 0;
  }

  if (!attack_fastest_opponent_as_defender()) {
    LOG_DEB(0, << "dont go to ball, (attack_fastest_op_as_def)");
    return false;
  }

  if(surpress_intercept()==true) {
    LOG_DEB(0, << "dont go to ball, (surpress_intercept)");
    return 0;
  }

  /* if its getting dangerous use blockin instead of intercepting (disabled by david 17.8)*/ 
  if(1 && intercepting_is_dangerous()){
    //LOG_INFO("Intercepting Disabled -> Blocking!!!");
    return block_ball_holder(aaction);
  }

  // ridi: save energy, if no chance to get ball
  // if you change this, also change code in pj_no_ball
  if((mdpInfo::get_stamina_state()==STAMINA_STATE_RESERVE) &&
     (WSinfo::ball->pos.getX() >20) &&
     ((steps2go.me > steps2go.opponent + 2) || steps2go.opponent == 0)){
    LOG_POL(4,"PJ go2ball: Stamina low, my steps to ball "<<steps2go.me
	    <<" op is faster "<<steps2go.opponent<<" ->FACE_BALL");
    LOG_ERR(0,"PJ go2ball: Stamina low, my steps to ball "<<steps2go.me
	    <<" op is faster "<<steps2go.opponent<<" ->FACE_BALL");
    LOG_DAN(0, << "Face ball because stamina is low!");
    set_aaction_face_ball_no_turn_neck(aaction);
    return true;
  }

  //LOG_DEB(0,"INTERCEPTING.....");
  return intercept_ball(aaction);
}

bool Noball03::test_tackle_aggressive(AAction &aaction) {
  double suc = Tools::get_tackle_success_probability(WSinfo::me->pos, WSinfo::ball->pos, WSinfo::me->ang.get_value());
  LOG_RIDI(0, << "Test Tackle: probability: " << suc);

  if (WSinfo::ws->play_mode != PM_PlayOn) {
    return 0;
  }
  if (WSinfo::is_ball_kickable() && (!steps2go.ball_kickable_for_opponent)) {
    LOG_RIDI(0, << "dont tackle, ball is kickable for me but not for him ");
    return 0;
  }
  if ((steps2go.me <= steps2go.opponent) && 
      (steps2go.me <= 2) &&  // I get the ball within the next two cycles
      (!steps2go.ball_kickable_for_opponent)) {
    return 0;
  }

  // compute if opponent has the ball
  bool opponent_has_ball = false;
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, ServerOptions::kickable_area); 
  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num > 0 )
    opponent_has_ball = true;

  float tackle_thresh = .75;

  if(opponent_has_ball == true && suc>tackle_thresh){
    Vector look_dir=Vector(WSinfo::me->ang);
    look_dir.normalize(10);//10 sinnvoll?
    Vector dir=Vector(WSinfo::me->ang);
    dir.normalize(1.5);//10 sinnvoll?
    Vector before_me=WSinfo::me->pos+look_dir;
    Quadrangle2d check_area_before_me=Quadrangle2d(WSinfo::me->pos+dir,before_me,2,6);
    Vector behind_me=WSinfo::me->pos-look_dir;
    Quadrangle2d check_area_behind_me=Quadrangle2d(WSinfo::me->pos-dir,behind_me,2,6);
    DBLOG_DRAW(0,check_area_before_me);
    DBLOG_DRAW(0,check_area_behind_me);
    PlayerSet opps_before=WSinfo::valid_opponents;
    opps_before.keep_players_in(check_area_before_me);
    PlayerSet opps_behind=WSinfo::valid_opponents;
    opps_behind.keep_players_in(check_area_behind_me);
    PlayerSet team_before=WSinfo::valid_teammates_without_me;
    team_before.keep_players_in(check_area_before_me);
    PlayerSet team_behind=WSinfo::valid_teammates_without_me;
    team_behind.keep_players_in(check_area_behind_me);
    double front_good=0.0;
    double back_good=0.0;
    
    if(before_me.getX()<-FIELD_BORDER_X+5 && fabs(before_me.getY())<10)
      front_good-=99.0;    
    if(behind_me.getX()<-FIELD_BORDER_X+5 && fabs(behind_me.getY())<10)
      back_good-=99.0;
    if (opps_before.num==0){
      if (team_before.num==0) {
        if((fabs(before_me.getX())>=FIELD_BORDER_X||fabs(before_me.getY())>=FIELD_BORDER_Y) && opps_before.num==0)
        front_good+=1.0;
      }
      else front_good += team_before.num;
    } else if (team_before.num==0) front_good += -opps_before.num;
    else{
      double dist_team=(team_before.closest_player_to_point(WSinfo::ball->pos))->pos.distance(WSinfo::ball->pos);
      double dist_opps=(opps_before.closest_player_to_point(WSinfo::ball->pos))->pos.distance(WSinfo::ball->pos);
      front_good+=(dist_opps-dist_team)/10;
    }
    if (opps_behind.num==0){
      if (team_behind.num==0) {
        if((fabs(behind_me.getX())>=FIELD_BORDER_X||fabs(behind_me.getY())>=FIELD_BORDER_Y) && opps_behind.num==0)
        back_good+=1.0;
      }
      else back_good += team_behind.num;
    } else if (team_behind.num==0) back_good += -opps_behind.num;
    else{
      double dist_team=(team_behind.closest_player_to_point(WSinfo::ball->pos))->pos.distance(WSinfo::ball->pos);
      double dist_opps=(opps_behind.closest_player_to_point(WSinfo::ball->pos))->pos.distance(WSinfo::ball->pos);
      back_good+=(dist_opps-dist_team)/10;
    }
    if (front_good<0 &&back_good<0) return false;
    else{
      if (front_good==back_good){ //prefer to shoot to his goal
        if(before_me.distance(Vector(FIELD_BORDER_X,0))<behind_me.distance(Vector(FIELD_BORDER_X,0)))
	  front_good+=1.0;
	else
	  back_good+=1.0;
        }
     if (front_good>=back_good) set_aaction_tackle(aaction, 100);
   else set_aaction_tackle(aaction, -100);
   }
    
    LOG_RIDI(0, << "Test Tackle: Opponent has ball and suc > threshold -> tackle ");
    return true;
  }

  LOG_RIDI(0, << "Test Tackle: NO tackle situation ");

  return false;
}







bool Noball03::test_tackle(AAction &aaction) {
  //LOG_POL(0, << "Time before test_tackle computation " << Tools::get_current_ms_time() - local_time);

  // ridi 04: code to tackle more often
  return test_tackle_aggressive(aaction);



  double suc = Tools::get_tackle_success_probability(WSinfo::me->pos, WSinfo::ball->pos, WSinfo::me->ang.get_value());
  LOG_RIDI(0, << "Test Tackle: probability: " << suc);
  //LOG_DEB(0, << "tackle success prob. = " << suc);
  /*
    if (!i_am_fastest_teammate()) {
    return NULL;
    }*/
  if (WSinfo::ws->play_mode != PM_PlayOn) {
    return 0;
  }
  if (WSinfo::is_ball_kickable()) {
    return 0;
  }
  if ((steps2go.me <= steps2go.opponent) && 
      (!steps2go.ball_kickable_for_opponent)) {
    return 0;
  }

  LOG_RIDI(0, << "Test Tackle: 1 ");
  
  if ((steps2go.opponent < 2) && 
      (suc > 0.95) &&
      steps2go.ball_kickable_for_opponent &&
      (my_role == 0) &&
      (((steps2go.opponent_pos.getY() > ServerOptions::goal_width/2.0 + 3.0)&&(WSinfo::me->ang.get_value() < PI)) ||
       ((steps2go.opponent_pos.getY() < -ServerOptions::goal_width/2.0 - 3.0)&&(WSinfo::me->ang.get_value() > PI))) ) {
    set_aaction_tackle(aaction, 100);
    return true;
  }

  LOG_RIDI(0, << "Test Tackle: 2 ");

  if ((steps2go.opponent < 2) && 
      //(steps2go.opponent_pos - WSinfo::me->pos).sqr_norm() < 1.0 &&
      ((WSinfo::me->ang.get_value() < PI/2.0) || (WSinfo::me->ang.get_value() > 1.5 * PI))) {
    if ((suc > 0.95) &&
	(my_role == 0)) {
      set_aaction_tackle(aaction, 100);
      return true;
    }

  LOG_RIDI(0, << "Test Tackle: 3 ");

    if ((suc > 0.5) && (my_role != 0) && RIGHT_PENALTY_AREA.inside(WSinfo::ball->pos)) {
      if (WSinfo::ball->pos.getY() > WSinfo::me->pos.getY()) {
	if (steps2go.ball_kickable_for_opponent &&
	    (mdpInfo::my_angle_to(Vector(52.5, ServerOptions::goal_width/2.0 - 1.0)) > 5.0*M_PI/180.0) &&
	    (mdpInfo::my_angle_to(Vector(52.5, ServerOptions::goal_width/2.0 - 1.0)) < 355.0*M_PI/180.0)) {
	  //LOG_DEB(0, << "turning to goal (i want to tackle)");
	  
	  set_aaction_turn_inertia(aaction, mdpInfo::my_angle_to(Vector(52.5, ServerOptions::goal_width/2.0 - 1.0)));
	  return true;
	} 
      } 
      else {
	if (steps2go.ball_kickable_for_opponent &&
	    (mdpInfo::my_angle_to(Vector(52.5, -ServerOptions::goal_width/2.0 + 1.0)) > 5.0*M_PI/180.0) &&
	    (mdpInfo::my_angle_to(Vector(52.5, -ServerOptions::goal_width/2.0 + 1.0)) < 355.0*M_PI/180.0)) {
	  //LOG_DEB(0, << "turning to goal (i want to tackle)");

	  set_aaction_turn_inertia(aaction, mdpInfo::my_angle_to(Vector(52.5, -ServerOptions::goal_width/2.0 + 1.0))); 
	  return true;
	} 
      }
      set_aaction_tackle(aaction, 100);
      return true;
    }

  LOG_RIDI(0, << "Test Tackle: 4 ");

    if ((suc > 0.7) &&
	(my_role != 0)) {
      LOG_RIDI(0,"Tackling");
      set_aaction_tackle(aaction, 100);
      return true;
    }
  }

  LOG_RIDI(0, << "Test Tackle: NO tackle situation ");

  return false;
}

bool Noball03::test_help_blocking(AAction &aaction){
  //LOG_POL(0, << "Time before test_help_blocking computation " << Tools::get_current_ms_time() - local_time);


  //return NULL;



  // do not help blocking if we are in standard situation
  if(opponent_free_kick_situation() || my_free_kick_situation()){
    return 0;
  }

  //only defenders can help blocking
  if(my_role != 0){
    return 0;
  }

  if (steps2go.ball_kickable_for_teammate) {
    return 0;
  }

  //do not help blocking if the ball is not inside the penalty area
  Vector ball_pos = WSinfo::ball->pos;

  Vector opponent_pos = steps2go.opponent_pos;

  if ((ball_pos.getX() < last_player_line + 3.0) &&
      (opponent_pos.getX() > -ServerOptions::pitch_length/2.0 + 18.0) &&
      (opponent_pos.getX() < last_player_line + 5.0) &&
      (fabs(opponent_pos.getY()) < ServerOptions::penalty_area_width/2.0 - 5.0)) {
    LOG_DAN(0, << "Help blocking, opponent broke through(outside penalty area)!");


    //is interceptor in trouble?
    Vector interceptor = steps2go.teammate_pos;
    if (interceptor.getX() > opponent_pos.getX()) {
      
      // get my rank in the go2ball list
      int pos = 0;
      int i=0;
      while(i<22){
	if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM){
	  if(WSinfo::ws->my_goalie_number != go2ball_list[i].number){
	    if(WSinfo::me->number == go2ball_list[i].number){
	      //LOG_DEB(0, "Ich habe pos " << pos);
	      i = 22;
	    }
	    else{
	      //LOG_DEB(0, << "Spieler " << go2ball_list[i].number << " hat pos " << pos);
	      pos++;
	    }
	  }
	}
	i++;
      }	
      
      if (pos == 1) {
	LOG_DAN(0, << "Emergency BLOCKING.....");
	return block_ball_holder(aaction);
      }
    }

    return 0;
    

  }


  if ( (DeltaPositioning::get_role(steps2go.teammate_number) >= 1) &&
       ( last_player_line + 5.0 > steps2go.opponent_pos.getX() || steps2go.opponent_pos.getX() < -36.0 ) ) {
    LOG_DAN(0, << "Help blocking, opponent is closer to goal than the (midfielder or attacker) teammate");
  } else if ((ball_pos.getX() > -52.5+18.0) ||  //16.0
	     (fabs(ball_pos.getY()) > 12.0)) {  //16.0
	return 0;
  }

  
  Vector interceptor = Vector(0.0, 0.0);
  if (WSinfo::get_teammate_by_number(steps2go.teammate_number) != NULL) {
    interceptor = WSinfo::get_teammate_by_number(steps2go.teammate_number)->pos;
  }

  if (ball_pos.getX() < -52.5+5.0) {
    if (fabs(ball_pos.getY()) > fabs(interceptor.getY())) {
      //mdpInfo::teammate_pos_abs(steps2go.teammate_number).y)) {
    } else {
      return 0;
    }
  }
  
  //is interceptor in trouble?

  //mdpInfo::teammate_pos_abs(steps2go.teammate_number);
  Vector op_to_goal = ServerOptions::own_goal_pos - opponent_pos;
  Vector op_to_interceptor = interceptor - opponent_pos;
  if (op_to_goal.dot_product(op_to_interceptor) < 0.0 &&
      //(interceptor.x > opponent_pos.x) &&
      (fabs(opponent_pos.getY()) < 10.0) &&
      (opponent_pos.getX() < -ServerOptions::pitch_length/2.0 + 18.0) ){
    /* get my rank in the go2ball list */
    int pos = 0;
    int i=0;
    while(i<22){
      if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM){
	if(WSinfo::ws->my_goalie_number != go2ball_list[i].number){
	  if(WSinfo::me->number == go2ball_list[i].number){
	    i = 22;
	  }
	  else{
	    pos++;
	  }
	}
      }
      i++;
    }	

    if (pos == 1) {
      //ERROR_OUT << " cycle " << WSinfo::ws->time << " player " << WSinfo::me->number << " EMERGENCY BLOCKING ";
      LOG_DAN(0, << "Emergency BLOCKING.....");
      //return 0;
      return block_ball_holder(aaction);
    }
  }

  return 0;
}


bool Noball03::surpress_intercept(){
  /* intercepting can be dangerous for defenders only*/
  if (do_surpress_intercept == false)
    return false;
  if (my_role != 0) {
    return false;
  }
  Vector intercept_pos;
  int tmp_steps;
  Policy_Tools::intercept_min_time_and_pos_hetero(tmp_steps, intercept_pos, WSinfo::ball->pos, 
					   WSinfo::ball->vel, WSinfo::me->pos, 
					   WSinfo::me->number, true, 0.8, -1000.0); 
  if(intercept_pos.getX() > 0 && (steps2go.me + 2 >= steps2go.opponent)){
    //LOG_DEB(0,"Defender could intercept, but intercept pos too large "<<intercept_pos);
    return true;
  }
  if(intercept_pos.getX() > 5 && (steps2go.me + 4 >= steps2go.opponent)){
    //LOG_DEB(0,"Defender could intercept, but intercept pos too large "<<intercept_pos);
    return true;
  }
  return false;
}


/* this function indicates if intercepting is dangerous
   or senseless in the current situation. intercepting
   can be disabled an replaced by blocking.
*/
bool Noball03::intercepting_is_dangerous(){
  /* intercepting can be dangerous for defenders only*/
  if(my_role != 0) {
    LOG_DAN(0, << "my role is not defender, but " << my_role);
    LOG_DAN(0, << "my number is " << WSinfo::me->number);
    return false;
  }

  int my_steps = steps2go.me;

  PPlayer p_tmp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  if (p_tmp != NULL) {
    LOG_DAN(0, " opponent age is " << p_tmp->age);
  }

  /* intercepting is not dangerous if i am faster to the ball*/
  if (mdpInfo::get_my_intention() == DECISION_TYPE_BLOCKBALLHOLDER) {
    if (!LEFT_PENALTY_AREA.inside(WSinfo::ball->pos)) {
      LOG_DAN(0, "my intention is block ball holder!");
      if ( ((my_steps < steps2go.opponent-1) &&
	   (p_tmp != NULL) &&
	   (p_tmp->age < 1)) ||
	   ((my_steps < steps2go.opponent-2) &&
	   (p_tmp != NULL) &&
	   (p_tmp->age <= 1)) ||
	   (my_steps < steps2go.opponent-3) ) {
	return false;
      }
    } else {
      if ( p_tmp != NULL && p_tmp->age >= 1 ) {
	if (p_tmp->age >= 1) {
	  if(my_steps < steps2go.opponent-1)
	    return false;
	} else
	  if(my_steps < steps2go.opponent)
	    return false;
      }
      if(my_steps<=steps2go.opponent){
	return false;
      }
    }
    return true;
  }

  /* intercepting is senseless if the opponent is already
     in ball possession */
  if(steps2go.opponent < 2){
    //LOG_DEB(1,"intercepting dangerous: opponent has ball");
    return true;
  }

  /* intercepting is senseless if the opponent is obviously faster to the ball */
  /*
  if(steps2go.me > steps2go.opponent+5){
    //LOG_DEB(1,"intercepting dangerous: opponent is faster");
    return true;
    }*/


  if (my_steps<=steps2go.opponent) return false;
 
  Vector opponent_pos = steps2go.opponent_pos;
  Vector ball_pos = WSinfo::ball->pos;// Policy_Tools::next_intercept_pos();
  Vector my_pos = WSinfo::me->pos;
  Vector my_goal = Vector(-52.5,0);
  
  float op_goal_ang = (my_goal-opponent_pos).arg();
  float op_me_ang =  (my_pos-opponent_pos).arg();
  float op_ball_ang =  (ball_pos-opponent_pos).arg();
  float m_ang_diff = Tools::get_abs_angle(op_me_ang-op_goal_ang);
  float b_ang_diff = Tools::get_abs_angle(op_ball_ang-op_goal_ang);

  if(m_ang_diff>0.3*PI&&(b_ang_diff<0.4*PI||steps2go.opponent<2)){
    //ERROR_OUT << "intercepting dangerous: lot dist too high";
    LOG_DEB(0,"interecepting dangerous: bad angle for intercepting");
    return true;
  }


  Vector lot = Tools::get_Lotfuss(ball_pos, my_goal, my_pos);
  if(lot.distance(my_pos)>10&&b_ang_diff<0.4*PI){
    //ERROR_OUT << "intercepting dangerous: lot dist too high";
    LOG_DEB(0,"intercepting dangerous: lot dist too high");
    return true;
  }
  
  return false;
}


Vector Noball03::get_block_position_DANIEL(int &use_long_go2pos) {
  /* get position of opponent ball holder */

  Vector ball_pos = steps2go.opponent_pos;

  Vector opponent_pos = steps2go.opponent_pos;
  Vector my_pos = WSinfo::me->pos;
  Vector my_goal = Vector(-52.5,0);

   

  if ((steps2go.opponent > 0) && //>=2
      (next_i_pos.getX() < steps2go.opponent_pos.getX())) {
    ball_pos = next_i_pos;
  }

  LOG_DAN(0, << _2D << VC2D(next_i_pos,0.5,"#000000"));

  Vector op_to_goal;
  double my_size = -1000.0;
  Vector block_op_pos = opponent_pos;

  if (steps2go.opponent > 0) {
    LOG_DAN(0, << "considering next_i_pos");

    my_size = 0.9 * (steps2go.opponent-1);//D
    block_op_pos = next_i_pos;//D
      
  }

  Vector res_pos;
  int res_time;
  float op_goal_ang;

  PPlayer p2 = WSinfo::get_opponent_by_number(steps2go.opponent_number);

  Cone2d cone;
  if (p2 != NULL) {
    if (M_PI <= (my_goal-block_op_pos).arg())
      cone = Cone2d(p2->pos, Vector(-1.0, 0.0), my_goal-block_op_pos);
    else
      cone = Cone2d(p2->pos, my_goal-block_op_pos, Vector(-1.0, 0.0));
  }

  
  
  Vector test_vec;
  test_vec.init_polar(1.0, p2->ang);

  /*
  if (my_pos.x < opponent_pos.x) {
    XYRectangle2d rect(Vector(-33.0, 36.0), Vector(0.0, ServerOptions::penalty_area_width/2.0 - 5.0));
    LOG_DEB(0, << _2D << rect);
    }*/


  if ((my_pos.getX() < opponent_pos.getX()) &&
      (opponent_pos.getX() > -33.) &&
      (fabs(opponent_pos.getY()) > ServerOptions::penalty_area_width/2.0 - 5.0)) {
    if (cone.inside(p2->pos + test_vec)) {
      op_to_goal = test_vec;
      op_goal_ang = p2->ang.get_value();
    } else {
      op_goal_ang = M_PI;
      op_to_goal = Vector(-1.0, 0.0);
    }
  } else if (WSinfo::ball->pos.getX() < -42.0 && fabs(WSinfo::ball->pos.getY()) > 5.0) {
    op_to_goal = my_goal - block_op_pos;
    op_goal_ang = op_to_goal.arg();
  } else if ( p2 != NULL && 
	      WSinfo::is_ball_kickable_for(p2) && 
	      p2->age_ang <= 2) {
    if (!cone.inside(WSinfo::me->pos)) {
      if (cone.inside(p2->pos + test_vec)) {
	op_to_goal = test_vec;
	op_goal_ang = p2->ang.get_value();
      } else {
	if (test_vec.getY() * p2->pos.getY() >= 0.0) {
	  op_goal_ang = M_PI;
	  op_to_goal = Vector (-1.0, 0.0);
	} else {
	  op_to_goal = my_goal - block_op_pos;
	  op_goal_ang = op_to_goal.arg();
	}
      }
    } else {
      //op_goal_ang = (M_PI + (my_goal-block_op_pos).arg()) / 2.0;
      op_goal_ang = (my_goal-block_op_pos).arg();
      op_to_goal.init_polar(1.0, op_goal_ang);
    }
  } else {
    //op_to_goal = my_goal - opponent_pos;
    op_to_goal = my_goal - block_op_pos;
    op_goal_ang = op_to_goal.arg();
  }
  

  //op_to_goal = my_goal - block_op_pos;
  //op_goal_ang = op_to_goal.arg();

  LOG_DEB(0, << _2D << VL2D(block_op_pos, (block_op_pos + 5.0 * op_to_goal), "ffffff"));
  /*
  if ( fabs(opponent_pos.y) > fabs(my_pos.y) &&
       fabs(opponent_pos.y) > 19.0 &&
       opponent_pos.x < my_pos.x + 4.0 &&
       Tools::get_abs_angle((my_pos-block_op_pos).arg()-op_goal_ang) > DEG2RAD(30.0) ) {
    if (opponent_pos.y > 0.0)
      return Vector(-50.0, 18.0);
    else
      return Vector(-50.0, -18.0);
      }*/


  Policy_Tools::intercept_player(res_time, res_pos,
				 block_op_pos, 0.8, 
				 op_goal_ang, WSinfo::me->pos,
				 1.0, my_size);

  float op_me_ang =  (my_pos-opponent_pos).arg();
  
  float m_ang_diff = Tools::get_abs_angle(op_me_ang-op_goal_ang);

  float b_dist = m_ang_diff*3.0;//3.0

  /*
    PPlayer p2 = WSinfo::get_opponent_by_number(steps2go.opponent_number);

    if ((p2->pos - WSinfo::me->pos).norm() < 1.0) {
    b_dist = m_ang_diff;
  } else if ((p2->pos - WSinfo::me->pos).norm() < 2.0) {
    b_dist = 0.5 + m_ang_diff * 2.0;
  } else {
    b_dist = 1.0 + m_ang_diff * 2.0;
  }

  if(b_dist < 1.0){
    b_dist = 0.2;
  }
  */

  LOG_DAN(0, << "b_dist = " << b_dist);
  //LOG_DAN(0, << _2D << C2D(res_pos.x,res_pos.y,0.5,"#ff0000"));
  op_to_goal.normalize(1.0);
  Vector res_pos2 = res_pos + b_dist * op_to_goal;

  PPlayer p_tmp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  
  if ( ((res_pos-opponent_pos).sqr_norm() < 1.0 &&
       m_ang_diff < DEG2RAD(90.0)) ||
       ((res_pos-opponent_pos).sqr_norm() < 4.0 &&
       m_ang_diff < DEG2RAD(30.0) &&
       p_tmp != NULL &&
       p_tmp->age_ang <= 1 &&
       Tools::get_abs_angle(op_goal_ang - (p_tmp->ang).get_value()) < DEG2RAD(20.0)) ) {

    if (((my_pos-opponent_pos).norm() > 1.2) || (m_ang_diff > DEG2RAD(40.0))) {
      res_pos2 = opponent_pos + 0.8 * op_to_goal;//0.5
    } else {
      res_pos2 = opponent_pos + 0.5 * op_to_goal;//0.3
    }
  }

  /*
  if (((res_pos2 - my_goal).norm() < 5.0) ||
      (res_pos2.x < -ServerOptions::pitch_length/2.0)) {
    if (res_pos2.x < -ServerOptions::pitch_length/2.0) {
      res_pos2 = block_op_pos;
    }
    }*/

  if (m_ang_diff < DEG2RAD(15.0)) {
    return res_pos2;
  }

  double norm1 = (res_pos2 - my_goal).norm();
  if ((norm1 < 12.0) ||
      //(norm1 < 18.0) && (sp < 0) ||
      (res_pos2.getX() < -ServerOptions::pitch_length/2.0)) {
    LOG_DAN(0, << "block position closer than 12.0 to goal, use special cases");
    double norm2 = (block_op_pos - my_goal).norm();
    if (norm2 < 8.0) {
      LOG_DAN(0, << "next_i_pos closer than 7.0, block at next_i_pos");
      res_pos2 = block_op_pos;
    } else if (norm2 < 10.0) {
      LOG_DAN(0, << "next_i_pos closer than 10.0, block at distance 5.0");
      op_to_goal = my_goal - block_op_pos;
      op_to_goal.normalize(1.0);
      res_pos2 = my_goal - 8.0*op_to_goal;
    } else if (norm2 < 12.5) {
      LOG_DAN(0, << "next_i_pos closer than 14.0, block at distance 8.0");
      op_to_goal = my_goal - block_op_pos;
      op_to_goal.normalize(1.0);
      res_pos2 = my_goal - 10.0*op_to_goal;
    } else {
      LOG_DAN(0, << "block at distance 12.0");
      op_to_goal = my_goal - block_op_pos;
      op_to_goal.normalize(1.0);
      res_pos2 = my_goal - 12.0*op_to_goal;
    }
  }

  return res_pos2;
}




/* blocking the opponent ball holder is done by going to a
   position between the opponent and our goal
*/
Vector Noball03::get_block_position(int &use_long_go2pos) {

  /* get position of opponent ball holder */
  
  Vector ball_pos = steps2go.opponent_pos;
  if ((steps2go.opponent > 0) && //>=2
      (next_i_pos.getX() < steps2go.opponent_pos.getX())) {
    ball_pos = next_i_pos;
  }

  Vector opponent_pos = steps2go.opponent_pos;
  Vector my_pos = WSinfo::me->pos;
  Vector my_goal = Vector(-52.5,0);

  float op_goal_ang;

  if ((my_pos.getX() < opponent_pos.getX()) &&
      (opponent_pos.getX() > -33.) &&
      (fabs(opponent_pos.getY()) > ServerOptions::penalty_area_width/2.0 - 5.0)) {
    op_goal_ang = M_PI;
  } else {
    op_goal_ang = (my_goal-opponent_pos).arg();
  }
  float op_me_ang =  (my_pos-opponent_pos).arg();
  
  float m_ang_diff = Tools::get_abs_angle(op_me_ang-op_goal_ang);

  float block_dist = 0.5 + m_ang_diff*5;

  PPlayer p = WSinfo::get_opponent_by_number(steps2go.opponent_number);

  if (p==NULL) return Vector(0.0,0.0);

  if (p->speed_max > 1.0) {
    block_dist = 0.5 + m_ang_diff*10;  
  }

  /*
  if (((p->pos - WSinfo::me->pos).norm() < 0.3) && 
      (m_ang_diff < 1.0)) {
    return WSinfo::me->pos;
    }*/

  if ((p->pos - WSinfo::me->pos).sqr_norm() < 1.0) {
    block_dist = m_ang_diff;
  } else if ((p->pos - WSinfo::me->pos).sqr_norm() < 4.0) { //2.0
    block_dist = 0.5 + m_ang_diff * 2;
  } else {
    block_dist = 1.0 + m_ang_diff * 5;
  }

  LOG_DAN(0, << "ang_diff = " << m_ang_diff);
  LOG_DAN(0, << "block_dist = " << block_dist);

  if(block_dist < 1.0){
    block_dist = 0.2;
  }
  //DANIEL
  Vector op2goal_line;

  if ((my_pos.getX() < opponent_pos.getX()) &&
      (opponent_pos.getX() > -33.) &&
      (fabs(opponent_pos.getY()) > ServerOptions::penalty_area_width/2.0 - 5.0)) {
    op2goal_line = Vector(-52.5, ball_pos.getY())-ball_pos;
  } else {
    op2goal_line = my_goal-ball_pos;
  }

  if(m_ang_diff > 0.4*PI){
    if(op2goal_line.norm() > 20.0)
      block_dist = op2goal_line.norm() - 13;
    else if(op2goal_line.norm() > 11.0)
      block_dist = op2goal_line.norm() - 7;
  }

  if (steps2go.opponent > 6) block_dist = 5.0;

  op2goal_line.normalize(block_dist);

  Vector target;

  target = ball_pos + op2goal_line;


  /*
  //if the opponent doesn't face the goal with his body, then move the block position in the direction he is facing
  //(where he'll probably run to)

  Angle op_ang_abs = mdpInfo::opponent_ang_abs(steps2go.opponent_number);
  int age_op_ang = mdpInfo::age_opponent_ang(steps2go.opponent_number);
  //if ((target-my_pos).norm() > 2) {

  if (!mdpInfo::is_ball_kickable_for_opponent(steps2go.opponent_number)) {
    //LOG_DEB(0, << "Ball is not kickable!");
  }
  if (block_dist <= 0.9) {
    //LOG_DEB(0, << "block_dist too small!");
  }
  if ((target-my_pos).norm() <= 2.0) {
    //LOG_DEB(0, << "target too close to my position!");
  }
  if (age_op_ang != 0) {
    //LOG_DEB(0, << "opponent angle too old!");
  }
  if (Tools::get_abs_angle((target-opponent_pos).arg() - op_ang_abs) <= DEG2RAD(20.0)) {
    //LOG_DEB(0, << "angle too small");
  }

  if ((mdpInfo::is_ball_kickable_for_opponent(steps2go.opponent_number)) && 
      //(block_dist > 0.9) && 
      ((target-my_pos).norm() > 2.0)) {
    if ((age_op_ang == 0) &&
	(Tools::get_abs_angle((target-opponent_pos).arg() - op_ang_abs) > DEG2RAD(20.0))) {
      //LOG_DEB(0, << "Player_ang seen " << mdpInfo::age_opponent_ang(steps2go.opponent_number) << " steps ago");
      //LOG_DEB(0, << "Changing block position");
      Vector add_vec = target - mdpInfo::opponent_pos_abs(steps2go.opponent_number);
      add_vec.normalize(1.0);
      if (op_ang_abs < PI) {
	add_vec.rotate(-PI/2);
      } else {
	add_vec.rotate(PI/2);
      }
      target += add_vec;
    }
  }
  */
  return target;
}




bool Noball03::block_ball_holder(AAction &aaction) {
  int use_long_go2pos = 0;
  Vector target = get_block_position_DANIEL(use_long_go2pos);
  //Vector target = get_block_position(use_long_go2pos);

  double left = -ServerOptions::pitch_length/2.0;
  double right = -ServerOptions::pitch_length/2.0 + 12.0;
  double top = 12.0;
  double bottom = -12.0;
  XYRectangle2d rect = XYRectangle2d(Vector(left, top), Vector(right, bottom));
  Vector own_goal_pos = Vector(-ServerOptions::pitch_length/2.0, 0.0);

  LOG_DEB(0, << _2D << rect);

  Line2d op_to_goal_line= Line2d(next_i_pos, (own_goal_pos - steps2go.opponent_pos));
  if (steps2go.opponent == 0)
    op_to_goal_line = Line2d(steps2go.opponent_pos, own_goal_pos - steps2go.opponent_pos);

  Line2d top_line(Vector(right, top-1.0), Vector(-1.0, 0.0));
  Line2d right_line(Vector(right-1.0, bottom), Vector(0.0, 1.0));
  Line2d bottom_line(Vector(left, bottom+1.0), Vector(1.0, 0.0));


  if ( WSinfo::me->stamina < 1700 && (steps2go.opponent_pos - WSinfo::me->pos).sqr_norm() > 6.25 ) {
    if (rect.inside(WSinfo::me->pos) &&
	!rect.inside(steps2go.opponent_pos)) {
      if (steps2go.opponent_pos.getX() < right &&
	  steps2go.opponent_pos.getY() > top) {
	if (!Geometry2d::intersect_lines(target, op_to_goal_line, top_line))
	  target = Vector(steps2go.opponent_pos.getX(), top-1.0);
	//target = point_on_line_y(own_goal_pos-steps2go.opponent_pos, steps2go.opponent_pos, top-1.0);
      } else if (steps2go.opponent_pos.getX() < right &&
		 steps2go.opponent_pos.getY() < bottom) {
	if (!Geometry2d::intersect_lines(target, op_to_goal_line, bottom_line))
	  target = Vector(steps2go.opponent_pos.getX(), bottom+1.0);
	//target = point_on_line_y(own_goal_pos-steps2go.opponent_pos, steps2go.opponent_pos, bottom+1.0);
      } else {
	if (!rect.inside(target)) {
	  if (!Geometry2d::intersect_lines(target, op_to_goal_line, right_line))
	    target = Vector(right-1.0, steps2go.opponent_pos.getY());
	  if (target.getY() > top) target.setY( top-1.0 );
	  else if (target.getY() < bottom) target.setY( bottom+1.0 );
	}
      }
    } else {
      left = -ServerOptions::pitch_length/2.0;
      right = -ServerOptions::pitch_length/2.0 + 15.0;
      top = 17.0;
      bottom = -17.0;
      rect = XYRectangle2d(Vector(left, top), Vector(right, bottom));
      top_line = Line2d(Vector(right, top-1.0), Vector(-1.0, 0.0));
      right_line = Line2d(Vector(right-1.0, bottom), Vector(0.0, 1.0));
      bottom_line = Line2d(Vector(left, bottom+1.0), Vector(1.0, 0.0));
  
      LOG_DEB(0, << _2D << rect);
      if (rect.inside(WSinfo::me->pos) &&
	  !rect.inside(steps2go.opponent_pos)) {
	if (steps2go.opponent_pos.getX() < right &&
	    steps2go.opponent_pos.getY() > top) {
	  if (!Geometry2d::intersect_lines(target, op_to_goal_line, top_line))
	    target = Vector(steps2go.opponent_pos.getX(), top-1.0);
	  //target = point_on_line_y(own_goal_pos-steps2go.opponent_pos, steps2go.opponent_pos, top-1.0);
	} else if (steps2go.opponent_pos.getX() < right &&
		   steps2go.opponent_pos.getY() < bottom) {
	  if (!Geometry2d::intersect_lines(target, op_to_goal_line, bottom_line))
	    target = Vector(steps2go.opponent_pos.getX(), bottom+1.0);
	  //target = point_on_line_y(own_goal_pos-steps2go.opponent_pos, steps2go.opponent_pos, bottom+1.0);
	} else {
	  if (!rect.inside(target)) {
	    if (!Geometry2d::intersect_lines(target, op_to_goal_line, right_line))
	      target = Vector(right-1.0, steps2go.opponent_pos.getY());
	    if (target.getY() > top) target.setY( top-1.0 );
	    else if (target.getY() < bottom) target.setY( bottom+1.0 );
	  }
	}
      }
    }
  }

  LOG_DAN(0, << _2D << C2D(target.x,target.y,0.5,"#00ff00"));
  LOG_DAN(1, << "Blocking opponent ball holder!");
  LOG_POL(3,"Blocking opponent ball holder");

  Vector my_pos= WSinfo::me->pos;
  
  LOG_DAN(0, << "Entfernung zum Ziel = " << (my_pos-target).norm());

  double me_to_target_sqr_norm = (my_pos-target).sqr_norm();

  if ((me_to_target_sqr_norm < 0.04) && //0.2
      ((my_pos-steps2go.opponent_pos).sqr_norm() < 1.0)) {
    set_aaction_turn_inertia(aaction, mdpInfo::my_angle_to(WSinfo::ball->pos));
    return true;
  }
  if (me_to_target_sqr_norm < 0.25) {//0.5
    mdpInfo::set_my_intention(DECISION_TYPE_BLOCKBALLHOLDER, target.getX(), target.getY());
    LOG_DAN(0, "very close, don't consider obstacles!");
    LOG_ERR(0,<<"very close, NeuroGo2Pos shoulnd't consider obstacles, but does!!!");
    set_aaction_goto(aaction, target, 0.2, 0, 1);
    return true;
  }
  if (me_to_target_sqr_norm < 4.0) {
    mdpInfo::set_my_intention(DECISION_TYPE_BLOCKBALLHOLDER, target.getX(), target.getY());
    Vector v1 = target - steps2go.opponent_pos;
    Vector v2 = my_pos - steps2go.opponent_pos;
    double sp = v1.getX()*v2.getX() + v1.getY() * v2.getY();
    if (sp >= 0.0) {
      LOG_DAN(0, "don't consider obstacles!");
      LOG_ERR(0,<<"very close, NeuroGo2Pos shoulnd't consider obstacles, but does!!!");
      set_aaction_goto(aaction, target, 0.3, 0, 1);
      return true;
    } else {
      LOG_DAN(0, "consider obstacles!");
      set_aaction_goto(aaction, target, 0.3, 1, 1);
      return true;
    }
  }
  mdpInfo::set_my_intention(DECISION_TYPE_BLOCKBALLHOLDER, target.getX(), target.getY());
  mdpInfo::set_my_neck_intention(NECK_INTENTION_BLOCKBALLHOLDER);
  if (steps2go.opponent >= 6)
    set_aaction_goto(aaction, target, (my_pos-target).norm() * 0.1, 1, 1);//0.5);
  else
    set_aaction_goto(aaction, target, (my_pos-target).norm() * 0.5, 1, 1);//0.5);
  return true;
}


bool Noball03::intercept_ball(AAction &aaction) {
  if((WSinfo::me->pos.distance(WSinfo::ball->pos)>25.0)&&(WSinfo::ball->vel.sqr_norm()>4.0)){ //2.0
    /* if ball is very fare away, go to the ball by Intercept_Ball(...) */
    LOG_POL(3, << "I am the fastest, ball is far away: Use Intercept_Ball(...)");
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
    LOG_POL(4,"I am the fastest 2 ball, Ball too fast and distance too far, use normal intercept!");
    if (my_role == 0) {
      mdpInfo::set_my_neck_intention(NECK_INTENTION_BLOCKBALLHOLDER);
    }
    set_aaction_go2ball(aaction);
    return true;
  }
  
  if((WSinfo::ball->vel.norm() <= 0.07) && (WSinfo::me->pos.distance(WSinfo::ball->pos) > 2.0)){
    /* if ball is not in move, go to the ball by Neuro_Go2Pos(...) */
    Vector ball_pos = WSinfo::ball->pos;
    LOG_POL(3, << "I am the fastest. Ball is slow: Use Neuro_Go2Pos(...)");
    LOG_POL(4,"I am the fastest 2 ball, Ball too slow and distance >2, use go2pos!");
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTSLOWBALL);
    /*
    if (my_role == 0) {
      mdpInfo::set_my_neck_intention(NECK_INTENTION_BLOCKBALLHOLDER);
      }*/
    set_aaction_goto(aaction, ball_pos, 0.5);
    return true;
  }
  
  LOG_POL(3, << "I am the fastest. Use Neuro_Intercept_Ball");
  LOG_POL(4,"I am the fastest 2 ball, use neurointercept!");
  mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
  /*
  if (my_role == 0) {
    mdpInfo::set_my_neck_intention(NECK_INTENTION_BLOCKBALLHOLDER);
    }*/

  set_aaction_go2ball(aaction);
  return true;
}

bool Noball03::opponent_free_kick_situation(){

  if(WSinfo::ws->play_mode == PM_my_BeforeKickOff
     || WSinfo::ws->play_mode == PM_my_AfterGoal
     || WSinfo::ws->play_mode == PM_his_BeforeKickOff
     || WSinfo::ws->play_mode == PM_his_AfterGoal
     || WSinfo::ws->play_mode == PM_his_KickOff
     || WSinfo::ws->play_mode == PM_his_KickIn
     || WSinfo::ws->play_mode == PM_his_FreeKick
     || WSinfo::ws->play_mode == PM_his_GoalKick
     || WSinfo::ws->play_mode == PM_his_GoalieFreeKick
     || WSinfo::ws->play_mode == PM_his_CornerKick
     || WSinfo::ws->play_mode == PM_my_GoalieFreeKick
     || WSinfo::ws->play_mode == PM_my_GoalKick) {
    return true;
  }
  
  return false;
}

bool Noball03::my_free_kick_situation() {
  if(WSinfo::ws->play_mode == PM_my_KickIn
     || WSinfo::ws->play_mode == PM_my_FreeKick
     || WSinfo::ws->play_mode == PM_my_CornerKick
     ){
    return true;
  }
  
  return false;
}

int Noball03::kick_in_by_midfielder(){
  /* Es sollte immer ein Mittelfeld-Spieler zum Ball gehen! */

  /* determine the fastest midfielder to intercept the ball */
  int gonumber = -1;
  for(int i=0;i<22;i++){
    if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM){
      if(DeltaPositioning::get_role(go2ball_list[i].number) == 1) {
	gonumber=go2ball_list[i].number;
	break;  
      }
    }
  }
  return gonumber;
}


int Noball03::kick_in_by_formation(){
  /* the player should do the kick in job which home position is 
     nearest to the kick in 
  */
  //D DeltaPositioning::update();

  time_of_last_kickin++;
  if(time_of_last_kickin>130 && i_am_fastest_teammate()) return WSinfo::me->number;

  double min_dist = 120;
  double min_dist2 = 120;
  int min_dist_player = -1;
  int min_dist2_player = -1;
  for(int i=2;i<=11;i++){
    Vector home_pos = DeltaPositioning::get_position(i);
    double dist = home_pos.distance(WSinfo::ball->pos);
    if(dist < min_dist){
      min_dist2 = min_dist;
      min_dist2_player = min_dist_player;
      min_dist = dist;
      min_dist_player = i;
    }
    if(dist < min_dist2 && dist > min_dist){
      min_dist2 = dist;
      min_dist2_player = i;
    }
  }
  
  //  if(mdpInfo::mdp->me->number == min_dist_player ||
  //     mdpInfo::mdp->me->number == min_dist2_player){
  //    cout << "\nPlayer: " << mdpInfo::mdp->me->number << " Min1:" << min_dist_player << " Dist1: " << min_dist << " Min2:" << min_dist2_player << " Dist2: " << min_dist2 << " Offside: " << DeltaPositioning::get_my_offence_line();
  //  }

  if(min_dist_player == -1){
    /*no player found? */
    return 0;
  }
  
  if(min_dist2_player == -1){
    return min_dist_player;
  }

  if(min_dist2-3 < min_dist){
    //der Unterschied betraegt nur 3 Meter, evtl. Weltmodell-Probleme
    //dann soll derjenige zum Ball gehen, der weiter vorn steht
    Vector homepos1 = DeltaPositioning::get_position(min_dist_player);
    Vector homepos2 = DeltaPositioning::get_position(min_dist2_player);
    if(homepos1.getX() < homepos2.getX()){
      return min_dist_player;
    } else{
      return min_dist2_player;
    }
  } else{
    return min_dist_player;
  }
  return 0;
}

bool Noball03::kick_in(AAction &aaction, int player_to_go){
  //wenn der Spieler ein Verteidiger ist, dann pruefe erst, 
  //ob Anzahl der Verteidiger ausreicht
  if(DeltaPositioning::get_role(player_to_go)==0){
    if(DeltaPositioning::get_num_defenders()<=3) player_to_go = kick_in_by_midfielder();
  }

  /* testif its my job to do the kick in? */
  if(player_to_go != WSinfo::me->number){
    return 0;
  }

  //wenn Ball sich im 16-Meter-Raum befindet, dann uebernimmt der Goalie den Part
  //if(mdpInfo::is_object_in_my_penalty_area(mdpInfo::ball_pos_abs())){ 
  //  return NULL;
  //}

  set_aaction_goto(aaction, WSinfo::ball->pos, 0.5);
  return true;
  //Vector ballpos = mdpInfo::ball_pos_abs();
}

/* ------------------------------------------------------------------- 
here ends code for the test_o2ball() stuff
---------------------------------------------------------------------*/


/** test_offside()
    if player is in offside this function will initiate a Neuro_Go2Pos 
    move to disable offside position
*/
bool Noball03::test_offside(AAction &aaction) {
  //LOG_POL(0, << "Time before test_offside computation " << Tools::get_current_ms_time() - local_time);

  int offside_tolerance = 1;

  if (DeltaPositioning::get_role(WSinfo::me->number) == PT_DEFENDER) return false;

  if (WSinfo::me->pos.getX() > DeltaPositioning::get_my_offence_line() - offside_tolerance){
    LOG_POL(3, << "Disable Offside. offside_line: " << DeltaPositioning::get_my_offence_line());
    /*
    LOG_ERR(0, << "PVQ NO BALL: I'm offside! my.x= "<<mdpInfo::mdp->me->pos_x.v<<" offside_line "
        <<DeltaPositioning::get_my_offence_line());
    */
    mdpInfo::set_my_intention(DECISION_TYPE_LEAVE_OFFSIDE,
			      (DeltaPositioning::get_my_offence_line()-offside_tolerance));

    // go one cycle to a position behind my current position
    set_aaction_goto(aaction, Vector(WSinfo::me->pos.getX() - 10.0, WSinfo::me->pos.getY()));
    return true;
  }
  return false;
}


/** test_look2ball()
    initiated default move if all test_... functions return null
*/
bool Noball03::test_look2ball(AAction &aaction) {
  /*
    if ((cover_number > 0) && (cover_number_valid_at == WSinfo::ws->time) && 
    (last_look_to_opponent < WSinfo::ws->time-3)) {
    PPlayer p_tmp = WSinfo::get_opponent_by_number(cover_number);
    if (p_tmp != NULL) {
    LOG_DAN(0, << "Look to opponnent because nothing to do");
    ANGLE dir = Tools::my_abs_angle_to(p_tmp->pos);
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, dir);
    last_look_to_opponent = WSinfo::ws->time;
    }
    } 

    if ((cover_number >= 0) && (cover_number_valid_at == WSinfo::ws->time)) {
    LOG_DAN(0, << "Turn to goal because nothing to do");
    ANGLE desired_dir = Tools::my_angle_to(Vector(-52.5, 0.0));
    set_aaction_turn_inertia(aaction, desired_dir.get_value());
    return true;
    }*/

  LOG_POL(3, << "Default Move: Face Ball");
  mdpInfo::set_my_intention(DECISION_TYPE_FACE_BALL);
  LOG_DAN(0, << "Face Ball because nothing to do");
  set_aaction_face_ball_no_turn_neck(aaction);
  return true;
}


bool Noball03::test_formations(AAction &aaction) {
  DashPosition my_form = DeltaPositioning::get_position(WSinfo::me->number);

  /* Entscheide je nach Rolle die Geschwindigkeit der Bewegung */
  //my_form.dash_power = stamina; 
  my_form.dash_power = 100;


  LOG_DAN(1, << "Positioning Target: "<< my_form.x << " " << my_form.y 
	     << " Dash allowed "<< stamina
	     << " Role "<< my_role);
  DashPosition my_fine = my_form;
  static int save_stamina = 0;
  if (fine_positioning) {
    if ( 0 == my_role )
      my_fine = positioning_for_defence_player( my_form );
    if ( 1 == my_role ) {
      my_fine = positioning_for_middlefield_player( my_form );
      LOG_RIDI(0, << "got back from postitonign ");
      if ((WSinfo::me->pos.getX() < my_fine.getX() + 10.0) &&
	  (fabs(WSinfo::me->pos.getY()-my_fine.getY()) < 10.0) &&
	  (WSmemory::team_last_at_ball() == 0) &&
	  (WSinfo::ball->pos.getX() < 0.0) ) {
	if (WSinfo::me->stamina < 2400 || (WSinfo::me->stamina < 3200 && save_stamina)) {
	  set_aaction_face_ball_no_turn_neck(aaction);
	  LOG_RIDI(0, << " SAVE STAMINA SAVE STAMINA SAVE STAMINA -> FACE BALL ");
	  save_stamina = 1;
	  return true;
	}
      }
      save_stamina = 0;
    }
    if ( 2 == my_role )
      my_fine = positioning_for_offence_player( my_form );
  }

  //bei bestimmten PlayModes darf der Spieler nicht durch die verbotenen Kreise laufen
  if(   WSinfo::ws->play_mode == PM_his_GoalKick
     || WSinfo::ws->play_mode == PM_his_GoalieFreeKick
     || WSinfo::ws->play_mode == PM_his_CornerKick
     || WSinfo::ws->play_mode == PM_his_FreeKick
     || WSinfo::ws->play_mode == PM_his_BeforeKickOff
     || WSinfo::ws->play_mode == PM_his_AfterGoal
     || WSinfo::ws->play_mode == PM_his_KickOff
     || WSinfo::ws->play_mode == PM_his_KickIn){
    Vector lf = Tools::get_Lotfuss(WSinfo::me->pos, my_fine, WSinfo::ball->pos);
    if ((lf.distance(WSinfo::ball->pos) < 9.15 && 
        lf.distance(my_fine) < WSinfo::me->pos.distance(my_fine) && 
        lf.distance(WSinfo::me->pos) < my_fine.distance(WSinfo::me->pos)) ||
       my_fine.distance(WSinfo::ball->pos) < 9.15){
      //ich darf mich nicht bewegen, da ich durch verbotenen Kreis laufen muesste
      if ((WSinfo::ws->play_mode == PM_his_FreeKick) ||
	  (WSinfo::ws->play_mode == PM_his_KickIn)) {
	Vector umgehung = 10 / (lf - WSinfo::ball->pos).norm() * (lf - WSinfo::ball->pos);
	umgehung +=  WSinfo::ball->pos;
	my_fine.clone( umgehung );
      } else {
	return 0;
      }
    }
  }

  //bei bestimmten PlayModes soll der Spieler Stamina sparen
  if (my_free_kick_situation()){
    if ((WSinfo::ws->play_mode ==  PM_my_GoalieFreeKick) &&
	(WSinfo::me->stamina >= ServerOptions::stamina_max - 500)) {
	/* D
	       (WSinfo::ws->time - DeltaPositioning::cycles_time_stamp >=  Goalie_Kickoff_Policy::wait_after_catch) &&
	       (((WSinfo::ws->time % Goalie_Kickoff_Policy::kickoff_at_modulo) >= 
		 Goalie_Kickoff_Policy::kickoff_at_modulo - 7) ||
		((WSinfo::ws->time % Goalie_Kickoff_Policy::kickoff_at_modulo) <= 2))) {
	*/
      my_fine.dash_power = 100;
    } else if (WSinfo::me->stamina < ServerOptions::stamina_max) {
      my_fine.dash_power = ServerOptions::stamina_inc_max - 10;
    } else {
      my_fine.dash_power = ServerOptions::stamina_inc_max;
    }
  }

  LOG_RIDI(0, << "OK, lets potio ");

#if 0
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_form.x, my_form.y ,1,"blue"));  // Homeposition
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_fine.x, my_fine.y ,1,"red")); // fineposition
  
#endif

  // Reached target area!
  if(my_fine.dash_power == 0.0){
    LOG_RIDI(0, << "my fine dash power is 0 ");
    return 0;
  }
  LOG_RIDI(0, << "OK, still here ");

  if (( 1 == my_role || 2 == my_role ) &&
	(WSinfo::me->pos.distance(Vector(my_fine)) < 2.0)){ // close enough
    LOG_POL(0,<<"Test formations: close enough to target face ball ");
    set_aaction_face_ball_no_turn_neck(aaction);
    return true;
  }
  LOG_RIDI(0, << "OK, still here ");


  LOG_POL(0, << "Fine Positioning Target: "<< my_fine.getX() << " " << my_fine.getY() << " Dash "<<my_fine.dash_power);

  //LOG_POL(0, << "End of test_formations reached after" << Tools::get_current_ms_time() - local_time << "millis");

  //just remember last go2pos movement to make fine positioning more continuoes!!!
  time_of_last_go2pos= WSinfo::ws->time;
  pos_of_last_go2pos= Vector(my_fine);

  LOG_RIDI(0, << "OK, still here final ");

  //LOG_ERR(0, << "can't use neuro_go2pos_stamina in test_formations => waste of stamina!!!");
  if ( my_role == 0) 
      set_aaction_goto(aaction, my_fine, 0.5, 1, 1);
  else
    set_aaction_goto(aaction, my_fine, 0.5);
  return true;
}


/* DANIEL DANIEL DANIEL defense stuff
   fine positioning for defenders
 */
bool  Noball03::test_cover_attacker(DashPosition & pos){
  

  /*dont cover opponents if my goalie is in ball possession */
  if(steps2go.my_goalie < 2){
    return false;
  }
  
  if(next_i_pos.getX() > 0.0){
    return false;
  }

  if ((next_i_pos.getX() > -30.0) &&
      (mdpInfo::is_my_team_attacking())) {
    LOG_DAN(0, << "Don't cover, our team is attacking!");
    return false;
  }
  
  char buffer[256];
  sprintf(buffer,"Go2Ball Analysis: Steps to intercept (Me %.2d) (Teammate %d %.2d) (Opponent %d %.2d)", 
	  steps2go.me, steps2go.teammate_number, steps2go.teammate,  steps2go.opponent_number,  
	  steps2go.opponent);  
  LOG_POL(0, << buffer);
  /* determine the attacker i have to take care of */ 

  Cover_Position c_pos;

  get_attacker_to_cover1(pos, c_pos);

  /*dont cover if i am still far away from my home position*/
  //DANIEL
  /*
  if(mdpInfo::my_pos_abs().x > pos.x+5) {
    //LOG_DEB(0, << "shoudln't cover: too far from homepos!");
    return false;
    }*/

  cover_number = c_pos.number;
  cover_number_valid_at = WSinfo::ws->time;

  /* no attacker to cover available*/ 
  if(c_pos.number <  0){
    Vector op_pos = next_i_pos;
    LOG_DAN(0, << "no Attacker to cover available");
    if ((op_pos.getX() < -40.0) && (op_pos.getX() > -47.0) &&
	(op_pos.getY() < 8.0) && (op_pos.getY() > -8.0) &&
	(steps2go.opponent <= steps2go.teammate)) {
      Vector corner1 = Vector(-ServerOptions::pitch_length/2.0, ServerOptions::goal_width/2.0);
      Vector corner2 = Vector(corner1.getX(), -corner2.getY());
      Vector shot_pos1 = point_on_line(op_pos-corner1, op_pos, -47.0);
      Vector shot_pos2 = point_on_line(op_pos-corner2, op_pos, -47.0);

      PlayerSet pset_tmp = WSinfo::valid_teammates;
      PPlayer p_tmp = NULL;
      int p1 = -1, p2 = -1;

      p_tmp = pset_tmp.closest_player_to_point(shot_pos1);
      if (p_tmp != NULL) p1 = p_tmp ->number;

      p_tmp = NULL;
      p_tmp = pset_tmp.closest_player_to_point(shot_pos2);
      if (p_tmp != NULL) p2 = p_tmp ->number;

      if (p1 == WSinfo::me->number) {
	pos.clone( shot_pos1 );
	LOG_DAN(0, << "Covering shoot line!");
	return true;
      } else if (p2 == WSinfo::me->number) {
	pos.clone( shot_pos2 );
	LOG_DAN(0, << "Covering shoot line!");
	return true;
      }
    }
    return false;
  }

  //if (DeltaPositioning::get_my_defence_line() <= ServerOptions::penalty_area_length) {

  //Vector cover_pos = get_cover_position(pos, attacker_number);
  Vector cover_pos = get_cover_position_DANIEL(pos, c_pos);

  //}
    
  LOG_DAN(0, << "c_pos = " << c_pos.pos.x << "," << c_pos.pos.y);

  LOG_DAN(0, << "Covering Attacker" << c_pos.number << "cover position (" 
	     << cover_pos.x << "," << cover_pos.y << ")");

  PPlayer p_tmp = WSinfo::get_opponent_by_number(c_pos.number);
  Vector op_pos(0,0);
  if (p_tmp != NULL) {
    op_pos = p_tmp->pos;
    if (p_tmp->age >= 1) {
      LOG_DEB(0, << "neck request LOOK TO COVERED OPPONENT set! ");
      Blackboard::set_neck_request(NECK_REQ_LOOKINDIRECTION, (op_pos-WSinfo::me->pos).arg());
    }
  }

  LOG_DAN(0, << "Attacker Pos" << op_pos.x 
	     << "," << op_pos.y);
  pos.clone( cover_pos );

  return true;
}

bool  Noball03::test_blocking_ball_holder(DashPosition & pos){

  if(steps2go.opponent>3) {
    return false;
  }
  
  if(pos.distance(WSinfo::ball->pos) > pos.radius){
    return false;
  }

  Vector op_pos = steps2go.opponent_pos;
  Vector goal_line = Vector(-52.5,0)-op_pos;
  for(float d=1.0;d<5.0;d+=1.0){
    goal_line.normalize(d);
    Vector block_pos = op_pos+goal_line;
    if(pos.distance(block_pos)<pos.radius){
      //LOG_INFO("HELP BLOCKING!");
      pos.clone( block_pos );
      return true;
    }
  }
  
  return false;
}

bool  Noball03::test_save_stamina(DashPosition & pos){
  /* no need to save stamina if theres enough available */
  if(WSinfo::me->stamina > 3000){
    return false;
  }
 
  /* if theres no danger at all, just to nothing to save stamina */
  Vector exp_ball_pos = next_i_pos;
  double ball_offset = exp_ball_pos.getX()-last_player_line;
  Vector my_pos = WSinfo::me->pos;

  if ((ball_offset > 30) && (last_player_line > -10)) {
    if(my_pos.distance(pos) < 8){
      pos.clone( my_pos );
      pos.dash_power = 0;
      return true;
    }
  }


  return false;
}

bool  Noball03::test_save_stamina_wm2001(DashPosition & pos){
  /* no need to save stamina if theres enough available */
  Vector my_pos= WSinfo::me->pos;

  if (pos.getX() < my_pos.getX()) //home pos is nearer to the goal then me
    return false;

  double stamina = WSinfo::me->stamina;
  double stamina_min_level = ServerOptions::recover_dec_thr * ServerOptions::stamina_max;
  double usable_stamina = stamina- stamina_min_level;

  //double stamina_per_meter = 100 - ServerOptions::stamina_inc_max;
  double stamina_per_meter = WSinfo::me->stamina_demand_per_meter;
  if (stamina_per_meter < 0)
    stamina_per_meter= 0;
  double max_dist_to_go= fabs( - ServerOptions::pitch_length*0.5  + 5.0 -  my_pos.getX() );
    
  if( max_dist_to_go * stamina_per_meter > usable_stamina) {
    //cerr << "\nsaving stamina, stamina= " << stamina << ", usable_stamina= " << usable_stamina << ", max_dist_to_go= " << max_dist_to_go << ", stamina/m= " << stamina_per_meter << ", num= " << mdpInfo::mdp->me->number << ", pos= " << my_pos;
    
    pos.setX( my_pos.getX() );
    pos.setY( my_pos.getY() );
    pos.dash_power = 0;
    return true;
  }

  return false;
}

bool Noball03::test_disable_moveup(DashPosition & pos){
  Vector my_pos = WSinfo::me->pos;
  if(my_pos.getX() < pos.getX()){
    /* dont move up if last player line is behind me */
    if(my_pos.getX() > last_player_line+5){
      pos.setX( my_pos.getX() );
      return true;
    }

#if 0 //this code is WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG 
    //dont move up if there is not enough stamina left
    if(WSinfo::me->stamina < 2000){
      return true;
    }
 
    /* dont move up if theres not enough syamina left to run back */
    if(my_pos.x > mdpInfo::stamina4meters_if_dashing(100)+stamina_run_back_line){
       pos.x = my_pos.x;
       pos.y = my_pos.y;

       pos.dash_power = 0;
       LOG_POL(4,"Dont move up because no stamina left to rush back!");
      return true;
    }
#endif
  }

  return false;
}

double Noball03::get_opponent_danger(int opponent_number) {
  double sum = 0;
  const double y_factor = 1.0;
  //const double ball_factor = 2.0;
  const double x_factor = 2.0;
  // consider y_position of opponent
  sum += y_factor * fabs(steps2go.opponent_pos.getY() - WSinfo::ball->pos.getY());
  //sum += ball_factor * fabs(mdpInfo::opponent_pos_abs(opponent_number).y - mdpInfo::ball_pos_abs().y) 
  //  / ServerOptions::pitch_width;

  sum += x_factor * fabs(steps2go.opponent_pos.getX() - last_player_line);

  return sum;
}


void Noball03::get_opponent_attackers_in_midfield(int *attacker_number, int max_attackers, 
							     int op_ball_holder) {
  double *op_danger_values = new double[11];
  int *opponent = new int[11];
  //LOG_DEB(0, << _2D << L2D(DeltaPositioning::get_my_defence_line()+6,FIELD_BORDER_Y, 
  //DeltaPositioning::get_my_defence_line()+6,- FIELD_BORDER_Y, "#888888"));
  for(int i=1;i<=11;i++){
    opponent[i-1] = -1;
    /* ignore opponents with invalid infos */
    if (!mdpInfo::is_opponent_pos_valid(i)) {
      //LOG_DEB(0, << "Opponent " << i << "position not valid");
      continue;
    }
    Vector op_pos = mdpInfo::opponent_pos_abs(i);
    /* ignore opponent holding the ball (its the interceptors job) */
    if (i == op_ball_holder) {
      //LOG_DEB(0, << _2D << C2D(op_pos.x,op_pos.y,2.0,"#ff0000"));
      continue;
    }
    op_danger_values[i-1] = get_opponent_danger(i);
    if ((fabs(op_pos.getY())>10) && (op_pos.getY() * WSinfo::ball->pos.getY() < 0)) continue;
    if (fabs(op_pos.getX() - DeltaPositioning::get_my_defence_line()) > 6.0) continue;
    //if(op_pos.x > -52.5+18) continue;
    //if(op_pos.distance(Vector(-52.5,0)) > 30 ) continue;
    opponent[i-1] = i; 
  }

  
  /* (2) filter attackers from the opponent array and copy them into the attacker_number array 
     attackers are opponent players near to my own goal, take into account at most max_attackers
     opponents
   */

  /* init with no attackers found */
  int num_attackers=0;
  for(int i=0;i<max_attackers;i++){
    attacker_number[i] = -1;
  }

  /*search for max_attackers attackers*/
  for(int j=0;j<max_attackers;j++){
    /* search for nearest opponent to goal in opponent array*/
    double min_dist = 1000;
    int min_number_idx = -1;
    for(int i=0;i<11;i++){
      /* ignore empty elements of the opponent array */
      if(opponent[i] == -1) continue;
      if(op_danger_values[i] < min_dist){
	min_dist = op_danger_values[i];
	min_number_idx = i;
      }
    }

    /* found nearest attacker? */
    if(min_number_idx > -1){
      //LOG_POLICY << "Found Attacker "<<  opponent[min_number_idx];
      /* copy opponent number into  attacker_number array */
      attacker_number[num_attackers++] = opponent[min_number_idx];
      /* delete entry in opponent array */
      opponent[min_number_idx] = -1;
    }
    else{
      attacker_number[num_attackers++] = -1;
    }
  }
  
  // ridi01: removed memory leak
  delete op_danger_values;
  delete opponent;


}


int Noball03::get_opponent_attackers(Cover_Position *c_pos_Arr, int max_attackers, 
						 int op_ball_holder){
  /* (1) create an array with numbers all relevant opponent players
     and corresponding distances to my goal (opponent array) */
  double *dist2goal = new double[11];
  int *opponent = new int[11];

  XYRectangle2d rect;

  if (WSinfo::ball->pos.getX() > -36.0) {
    rect = XYRectangle2d(Vector(-52.5, 8.0), Vector(-52.5+16.5,-8.0));
    LOG_DEB(0, << _2D << rect);
  } else {
    double upper_y_border = 15.0;
    double lower_y_border = -15.0;
    /*
    if (WSinfo::ball->pos.y > 17.4) {
      upper_y_border = WSinfo::ball->pos.y - 2.4;
      if (upper_y_border > 20.0)
	upper_y_border = 20.0;
    } else if (WSinfo::ball->pos.y < -17.4) {
      lower_y_border = WSinfo::ball->pos.y + 2.4;
      if (lower_y_border < -20.0)
	lower_y_border = -20.0;
	}*/
    rect = XYRectangle2d(Vector(-52.5, upper_y_border), Vector(-52.5+16.5, lower_y_border));
    LOG_DEB(0, << _2D << rect);
  }

  for(int i=1;i<=11;i++){
    opponent[i-1] = -1;
    /* ignore opponents with invalid infos */
    if(!mdpInfo::is_opponent_pos_valid(i)) continue;
    /* ignore opponent holding the ball (its the interceptors job) */
    if(i == op_ball_holder) continue;
    Vector op_pos = mdpInfo::opponent_pos_abs(i);
    /* ignore opponents far away from our goal) */
    if (!rect.inside(op_pos))
      continue;
     /*
    if (WSinfo::ball->pos.x > -36.0) {
      if(fabs(op_pos.y) > 8.0) continue;
    } else {
      if(fabs(op_pos.y) > 15.0) continue;
    }
    if(op_pos.x > -52.5+16.5) continue;
     */

    if(op_pos.distance(Vector(-52.5,0.0)) > 30.0 ) continue;
    double goal_dist = op_pos.distance(Vector(-52.5, 0.0));
    dist2goal[i-1] = goal_dist;
    opponent[i-1] = i; 
    //LOG_POLICY << "Found Opponent "<<  i << " Dist: " << goal_dist;
  }
 

  /* (2) filter attackers from the opponent array and copy them into the attacker_number array 
     attackers are opponent players near to my own goal, take into account at most max_attackers
     opponents
   */

  /* init with no attackers found */
  int attacker_idx = 0;
  for(int i=0;i<max_attackers;i++){
    c_pos_Arr[i].number = -1;
  }

  int num_players = 0;

  /*search for max_attackers attackers*/
  for(int j=0;j<max_attackers;j++){
    /* search for nearest opponent to goal in opponent array*/
    double min_dist = 1000;
    int min_number_idx = -1;
    for(int i=0;i<11;i++){
      /* ignore empty elements of the opponent array */
      if(opponent[i] == -1) continue;
      if(dist2goal[i] < min_dist){
	min_dist = dist2goal[i];
	min_number_idx = i;
      }
    }

    /* found nearest attacker? */
    if(min_number_idx > -1){
      //LOG_POLICY << "Found Attacker "<<  opponent[min_number_idx];
      /* copy opponent number into  attacker_number array */
      c_pos_Arr[attacker_idx].player = 1;
      c_pos_Arr[attacker_idx].number = opponent[min_number_idx];
      attacker_idx++;
      num_players++;
      /* delete entry in opponent array */
      opponent[min_number_idx] = -1;
    }
    else{
      c_pos_Arr[attacker_idx++].number = -1;
    }
  }
  
  // ridi01: removed memory leak
  delete dist2goal;
  delete opponent;

  return num_players;
}


/* Berechnet den Schnittpunkt zweier Geraden */
Vector Noball03::intersection_point(Vector p1, 
					       Vector steigung1, 
					       Vector p2, 
					       Vector steigung2) {
  LOG_DAN(0, << "steigung1 = (" << steigung1.getX() << "," << steigung1.getY() << ")");
  LOG_DAN(0, << "steigung2 = (" << steigung2.getX() << "," << steigung2.getY() << ")");
  double x, y, m1, m2;
  if ((fabs(steigung1.getX()) < 0.0001) || (fabs(steigung2.getX()) < 0.0001)) {
    return Vector(-51.5, 0);
  }
  m1 = steigung1.getY()/steigung1.getX();
  m2 = steigung2.getY()/steigung2.getX();
  if (fabs(m1-m2) < 0.0001) return Vector(-42.0, 0.0);
  x = (p2.getY() - p1.getY() + p1.getX()*m1 - p2.getX()*m2) / (m1-m2);
  y = (x-p1.getX())*m1 + p1.getY();
  return Vector (x, y);
}


/* Berechnet die y-Koordinate Punktes auf der Linie, der die x-Koordinate x hat
 */
Vector Noball03::point_on_line(Vector steigung, Vector line_point, double x) {
  //steigung.normalize();
  if (steigung.getX() != 0.0) {
    steigung = (1.0/steigung.getX()) * steigung;
  }
  if (steigung.getX() > 0) {
    return (x - line_point.getX()) * steigung + line_point;
  }
  if (steigung.getX() < 0) {
    return (line_point.getX() - x) * steigung + line_point;
  }
  // Zur Sicherheit, duerfte aber nie eintreten
  return line_point;
} /* point_on_line */


void Noball03::get_positions_to_cover(Cover_Position *c_pos_Arr, int num_c_players, 
						 int num_defenders) {
  //GO2003
  return;
  
  if (num_c_players == num_defenders) return;

  if (WSinfo::ws->play_mode != PM_PlayOn) return;
  
  int c_pos_idx = num_c_players;

  /*
  if (WSinfo::is_ball_kickable_for(WSinfo::get_opponent_by_number(steps2go.opponent_number))) {
    next_i_pos = WSinfo::ball->pos;
    }*/

  Vector pos;

  //LOG_DEB(0, << "next_i_pos = (" << next_i_pos.x << "," << next_i_pos.y << ")");
  //LOG_DEB(0, << "WSinfo::ball->pos = (" << WSinfo::ball->pos.x << "," << WSinfo::ball->pos.y << ")");

  int added_pos = 0;
  if ((WSinfo::ball->pos.getX() < -ServerOptions::pitch_length/2.0 + ServerOptions::penalty_area_length + 2.0) &&
      (fabs(WSinfo::ball->pos.getY()) > ServerOptions::goal_width/2.0 + 2.0)) {
    if (WSinfo::ball->pos.getY() > 0.0) {
      pos = intersection_point(next_i_pos, Vector(-43.0, 0.0)-next_i_pos, 
				      Vector(-53.0, ServerOptions::goal_width/2.0), Vector(1.0, 0.0));
    } else {
      pos = intersection_point(next_i_pos, Vector(-41.5, 0.0)-next_i_pos, 
				      Vector(-53.0, -ServerOptions::goal_width/2.0), Vector(1.0, 0.0));
    }
    if (pos.getX() < -50.0) return;
    c_pos_Arr[c_pos_idx].player = 0;
    c_pos_Arr[c_pos_idx].number = 0;
    c_pos_Arr[c_pos_idx].static_position = 0;
    c_pos_Arr[c_pos_idx].pos = pos;
    added_pos = 1;
    LOG_POL(0, " added cover position (" << pos.getX() << "," << pos.getY() << ")");
  }

  Vector opponent_pos = steps2go.opponent_pos;
  if ((opponent_pos.getX() < -ServerOptions::pitch_length/2.0 + ServerOptions::penalty_area_length - 2.0) &&
      (fabs(opponent_pos.getY()) < 10.0) &&
      (WSinfo::ws->time >= time_of_last_chance+5) &&
      (WSinfo::ws->play_mode == PM_PlayOn) &&
      (steps2go.ball_kickable_for_opponent)) {
    if ((num_c_players + 1 < num_defenders) ||
	((num_c_players < num_defenders) && (added_pos == 0))) {
      Vector tmp = Vector(-ServerOptions::pitch_length/2.0 , 0.0) - opponent_pos;
      tmp.normalize(0.3);
      pos = opponent_pos - tmp;
      if (added_pos == 1) ++c_pos_idx;
      c_pos_Arr[c_pos_idx].player = 0;
      c_pos_Arr[c_pos_idx].number = 0;
      c_pos_Arr[c_pos_idx].static_position = 0;
      c_pos_Arr[c_pos_idx].pos = pos;
      LOG_POL(0, " opponent has ball for 5 cycles, help attack");
      LOG_POL(0, " added cover position (" << pos.getX() << "," << pos.getY() << ")");
    }
  }
}


/* determine the attacker i have to take care of 
   (1) try to find an attcker inside the range wrt. my 
   home position determned by the positioning object.

   (2) if that fails check the unresolved attackers (attackers 
   not matched to a defender by criterion (1))

*/ 
void Noball03::get_attacker_to_cover1(const DashPosition & pos, Cover_Position &cover_pos){
  int MAX_DEFENDERS = 5;
  int defenders[MAX_DEFENDERS]; 
  for(int i=0;i<MAX_DEFENDERS;i++) defenders[i] = -1;
  int num_defenders=0;
  int my_index = -1;

  for(int i=2;i<=11;i++) {
    //skip midfielders and attackers
    if(DeltaPositioning::get_role(i) != 0) continue;
    //skip the interceptor
    if(i == steps2go.teammate_number) continue;
    if(i == WSinfo::me->number){
      my_index = num_defenders;
    }
    defenders[num_defenders++] = i;
  }
  
  Cover_Position *c_pos_Arr = new Cover_Position[num_defenders];
  Vector *targets = new Vector[num_defenders];
  Vector *sources = new Vector[num_defenders];

  // ignore the opponent ball_holder, to take care of him is the interceptors' job
  int op_ball_holder = steps2go.opponent_number;

  // compute attacking opponents 
  //DANIEL
  //if (DeltaPositioning::get_my_defence_line() <= ServerOptions::penalty_area_length) {

  int num_c_players = get_opponent_attackers(c_pos_Arr, num_defenders, op_ball_holder);

    //} else {
    //get_opponent_attackers(attackers, num_defenders, op_ball_holder);
    //}
  

  char buffer4[512];
  sprintf(buffer4,"NUMDEF: %d ",num_defenders);
  

  if (num_c_players < num_defenders) {
    get_positions_to_cover(c_pos_Arr, num_c_players, num_defenders);
  }

  int num_c_pos = 0;
  PPlayer p_tmp;
  for(int i=0;i<num_defenders;i++){
    if(c_pos_Arr[i].number != -1) {
      if (c_pos_Arr[i].player == 0) {
	targets[i] = c_pos_Arr[i].pos;
      } else {
	p_tmp = WSinfo::get_opponent_by_number(c_pos_Arr[i].number);
	if (p_tmp != NULL) targets[i] = p_tmp->pos;
	else targets [i] = Vector(0,0);
	//targets[i] = mdpInfo::opponent_pos_abs(c_pos_Arr[i].number);
      }
      num_c_pos++;
    }
    
    p_tmp = WSinfo::get_teammate_by_number(defenders[i]);
    if (p_tmp != NULL) sources[i] = p_tmp->pos;
    else sources[i] = Vector(0,0);

    //sources[i] = mdpInfo::teammate_pos_abs(defenders[i]);

    //DANIEL
    //if (DeltaPositioning::get_my_defence_line() < -32.0) sources[i] = mdpInfo::teammate_pos_abs(defenders[i]);
    //else sources[i] = (Vector) DeltaPositioning::get_position(defenders[i]);
  }



  MatchPositions matcher;
  
  matcher.match(sources, num_defenders, targets, num_c_pos);
  int my_cover_idx = matcher.get_match_for(my_index);

  char buffer1[256];
  char buffer2[32];
  double min_ball_dist = 1000;
  sprintf(buffer1,"MT: (%.1f %.1f) %.2f Attacker Matching: ", pos.getX(),pos.getY(), pos.distance(WSinfo::me->pos));
  for(int i=0;i<num_defenders;i++){
    if(matcher.get_match_for(i) != -1){
      sprintf(buffer2,"D%d-A%d | ", defenders[i],  c_pos_Arr[matcher.get_match_for(i)].number);
      strcat(buffer1,buffer2);
      LOG_DAN(0, << _2D << C2D(targets[matcher.get_match_for(i)].x,targets[matcher.get_match_for(i)].y,2.0,"#0000ff"));
    }
    else {
      p_tmp = WSinfo::get_teammate_by_number(defenders[i]);
      Vector p_pos = Vector(0,0);
      if (p_tmp != NULL) p_pos = p_tmp->pos;

      if (min_ball_dist > WSinfo::ball->pos.distance(p_pos)) {
	min_ball_dist = WSinfo::ball->pos.distance(p_pos);
      }
    } 
  }
  LOG_DAN(0, << buffer1);

  



  if(my_cover_idx == -1){
    cover_pos.number = -1;
    //return -1;
  } else {
    cover_pos.static_position = c_pos_Arr[my_cover_idx].static_position;
    cover_pos.player = c_pos_Arr[my_cover_idx].player;
    cover_pos.number = c_pos_Arr[my_cover_idx].number;
    cover_pos.pos = c_pos_Arr[my_cover_idx].pos;
  }
  // ridi01: clean up!

  delete c_pos_Arr;
  delete targets;
  delete sources;

}


Vector Noball03::get_cover_position_DANIEL(const DashPosition &pos, 
						      const Cover_Position &cover_pos) {

  if (cover_pos.player == 0) {
    LOG_DAN(0, << "position not a player");
    LOG_DAN(0, << "return value = " << cover_pos.pos.x << "," << cover_pos.pos.y);
    return cover_pos.pos;
  }
  
  PPlayer p_tmp = WSinfo::get_opponent_by_number(cover_pos.number);
  Vector op_pos = Vector(0,0);
  if (p_tmp != NULL) op_pos = p_tmp->pos;
  //Vector op_pos = mdpInfo::opponent_pos_abs(cover_pos.number);

  Vector ball_pos =  WSinfo::ball->pos;
  if (!steps2go.ball_kickable_for_opponent) {
    ball_pos = next_i_pos;
  }
  double op_goal_ang = (Vector(-52.5,0.0)-op_pos).arg();
  double op_ball_ang = (ball_pos-op_pos).arg();
  
  double opt_ang = op_ball_ang;
  //double opt_ang = op_goal_ang;

  XYRectangle2d rect = XYRectangle2d(Vector(-45.0, 8.0), Vector(-41.0, -8.0));
  
  Vector offset = Vector(1.5, 0.0);
  Vector offset2 = Vector(-1.0, 0.0);
  //Vector offset2 = Vector(1.0, 0.0); GO2003
  offset.rotate(opt_ang);
  //offset2.rotate(op_goal_ang);
  
  /* GO2003
  p_tmp = WSinfo::get_opponent_by_number(cover_pos.number);
  if ((p_tmp != NULL) && (p_tmp->vel.x < 0.0)) {
    offset2 = Vector(2.0 * p_tmp->vel.x, 0.0);
    }*/
  /*
  if ((op_pos.x > DeltaPositioning::get_my_defence_line()) &&
      (ball_pos.x > DeltaPositioning::get_my_defence_line())) {
    return (Vector(DeltaPositioning::get_my_defence_line(), op_pos.y));
    }*/

  Vector ret(0.0, 0.0);

  if ( (WSinfo::me->pos - (op_pos+offset)).norm() > 5.0 ) {
    ret = op_pos+offset+2.0*offset2;
  } else if ( (WSinfo::me->pos - (op_pos+offset)).norm() > 2.5 ) {
    ret = op_pos+offset+offset2;
  } else {
    if (op_pos.getX() > next_i_pos.getX() && rect.inside(op_pos)) {
      LOG_DEB(0, << _2D << rect);
      ret = op_pos+offset+offset2;
    }
    else
      ret = op_pos+offset;//D+offset2;
  }
  
  Vector off;

  off.init_polar(1.2, op_goal_ang);

  Quadrangle2d quad(ball_pos, op_pos, op_pos + off, ball_pos + off);
  
  LOG_DEB(0, << _2D << quad);

  if (quad.inside(WSinfo::me->pos) && (WSinfo::me->pos - ret).norm() < 1.5)
    return WSinfo::me->pos;

  return ret;

}

/*
int Noball03::teammate_pos_closest_to(Vector pos) {
  double min_dist = 1000.0;
  int player_number = -1;
  for (int i = 2; i <= 11; i++) {
    //if ((DeltaPositioning::get_position(i) - pos).norm() <= min_dist) {
    if ((mdpInfo::teammate_pos_abs(i) - pos).norm() <= min_dist) {
      //min_dist = (DeltaPositioning::get_position(i) - pos).norm();
      min_dist = (mdpInfo::teammate_pos_abs(i) - pos).norm();
      player_number = i;
    }
  }
  return player_number;
}
*/
  
/* compute best position to cover the opponent attacker */ 
Vector  Noball03::get_cover_position(const DashPosition & pos, int attacker_number){
  PPlayer p_tmp = WSinfo::get_opponent_by_number(attacker_number);
  Vector op_pos = Vector(0,0);
  if (p_tmp != NULL) op_pos = p_tmp->pos;
  //Vector op_pos = mdpInfo::opponent_pos_abs(attacker_number);

  Vector ball_pos =  WSinfo::ball->pos;
  if (!steps2go.ball_kickable_for_opponent) {
    ball_pos = next_i_pos;
  }
  double op_goal_ang = (Vector(-52.5,0.0)-op_pos).arg();
  double op_ball_ang = (ball_pos-op_pos).arg();
  


  double ang1 = Tools::min(op_goal_ang, op_ball_ang);
  double ang2 = Tools::max(op_goal_ang, op_ball_ang);
  double ang_diff = ang2-ang1;
  
  double opt_ang = op_ball_ang;
  //double opt_ang = op_goal_ang;

  if(ang_diff < 0.7*PI){
    ;//double opt_ang = ang1+(ang_diff)/2;
  }
  
  Vector offset = Vector(4.0, 0.0);
  offset.rotate(opt_ang);

  return op_pos+offset;
}

Vector Noball03::ball_pos_abs_after_steps(int steps) {
  Vector b_pos = WSinfo::ball->pos;
  Vector b_vel = WSinfo::ball->vel;
  for (int i = 0; i < steps; i++) {
    b_pos += b_vel;
    b_vel /= ServerOptions::ball_decay;
  }
  return b_pos;
}

Vector Noball03::next_intercept_pos_NN() {
  if (steps2go.teammate < steps2go.opponent) {
    //LOG_DEB(0, << "steps: " << steps2go.teammate);
    return ball_pos_abs_after_steps((int)steps2go.teammate);
  } else {
    //LOG_DEB(0, << "steps: " << steps2go.opponent);
    return ball_pos_abs_after_steps((int)steps2go.opponent);
  }
}

