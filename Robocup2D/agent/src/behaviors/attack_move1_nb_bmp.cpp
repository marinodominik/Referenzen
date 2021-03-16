#include "attack_move1_nb_bmp.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "geometry2d.h"

#if 0
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"ATTACK_MOVE1_NB: " XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define MYGETTIME (0)
#endif

#define DIST2OFFSIDELINE 1.0

bool Attack_Move1_Nb::initialized=false;
bool Attack_Move1_Nb::activated=true;


#define LA_Y 5.0 // default Y for left attacker
#define CA_Y 0.0 // default Y for centre attacker
#define RA_Y -5.0 // default Y for right attacker

#define LM_Y 12.0 // default Y for left midfielder
#define CM_Y 0.0 // default Y for centre midfielder
#define RM_Y -12.0 // default Y for right midfielder


#define M_BEHIND_OFFSIDE_DEFAULT 15.
#define M_BEHIND_OFFSIDE_AGGRESSIVE 10.

#define MIDDLE 0
#define RIGHT 1
#define LEFT 2

/* constructor method */
Attack_Move1_Nb::Attack_Move1_Nb() {
  go2pos = new NeuroGo2Pos;
  face_ball = new FaceBall;
  basiccmd = new BasicCmd;
  y_variation = 4.; // could be made player dependend
  x_variation = 3.; // could be made player dependend
  mindist2teammate =4.0;

  ValueParser vp(CommandLineOptions::policy_conf,"attack_move1_nb");
  vp.get("activated",activated);
}

Attack_Move1_Nb::~Attack_Move1_Nb() {
  delete go2pos;
  delete face_ball;
  delete basiccmd;
}


bool Attack_Move1_Nb::do_move(){
  if (activated == false){
    DBLOG_POL(0, << "NOT ACTIVATED ");
    return false;
  }

  //  if(WSinfo::ball->pos.x< FIELD_BORDER_X - 25.){ // not an attack situation
  if(WSinfo::ball->pos.getX()<25.){ // not an attack situation
    DBLOG_POL(0, << "NOT MY SITUATION ");
    return false;
  }
  DBLOG_POL(0, << "HEY, ITS MY SITUATION!!! ");
  return true;
}

void Attack_Move1_Nb::determine_offside_line(){
  if(WSmemory::get_his_offsideline_movement() <0){
    DBLOG_POL(1,"offside line moves towards us, increase critical distance!");
    offside_line = WSinfo::his_team_pos_of_offside_line() - 2* DIST2OFFSIDELINE;
  }
  else
    offside_line = WSinfo::his_team_pos_of_offside_line() - DIST2OFFSIDELINE;
  if(offside_line < 0. - DIST2OFFSIDELINE){
    offside_line = -DIST2OFFSIDELINE;
  }
  if (offside_line > FIELD_BORDER_X - 2)
    offside_line = FIELD_BORDER_X - 2;
  DBLOG_DRAW(0,L2D( WSinfo::his_team_pos_of_offside_line() ,-30, WSinfo::his_team_pos_of_offside_line() ,30,"green"));
  DBLOG_DRAW(0,L2D( offside_line,-30, offside_line,30,"green"));
}

bool Attack_Move1_Nb::is_mypos_ok(const Vector & targetpos) {
  // determine whether to move or to stay
  Vector mypos = WSinfo::me->pos;   // targetposition; default: my current position
  float max_y_tolerance = 2; // could be made player dependend
  
  if(mypos.getX() <= targetpos.getX())  // too far behind
    return false;
  if(mypos.getY() > targetpos.getY() + max_y_tolerance) // too far left
    return false;
  if(mypos.getY() < targetpos.getY() - max_y_tolerance) // too far right
    return false;
  if(mypos.getX() >offside_line)
    return false;
  return true;
}


Vector Attack_Move1_Nb::get_targetpos_leftmidfielder() {
  //  Vector tp = Vector(offside_line - M_BEHIND_OFFSIDE_DEFAULT,LM_Y);
  float x_var = x_variation;
  float y_var = y_variation;
  float xdist2teammate = 5.0;
  float ydist2teammate = 12.0;  // ridi: was 10.0 -> experiment with this
  float xdefault = offside_line - M_BEHIND_OFFSIDE_DEFAULT;
  float ydefault = LM_Y;

  PPlayer teammate;
  WSinfo::get_teammate(7,teammate); // default teammate


  // default position:
  Vector tp = Vector(xdefault,ydefault);
  int prefer = MIDDLE;

  if(WSinfo::ball->pos.getY() <0){ // ball is  right
    if(teammate){
      tp = teammate->pos;
      tp.subFromX( xdist2teammate );
      tp.addToY( ydist2teammate );
      tp.setY( Tools::max(tp.getY(),-3.0) ); // do not go further than that
      tp.setX( Tools::max(xdefault, tp.getX()) );
    }
    else{ // default, if 7 is not known
      tp.setY( 0.0 );
    }
    prefer=LEFT;
  }
  else  if(WSinfo::ball->pos.getY() > ydefault){
    tp.setY( WSinfo::ball->pos.getY() ); // if ball is left of my homepos, stay on ballheight
    prefer=RIGHT;
    tp.setX( Tools::max(WSinfo::ball->pos.getX() - 10., xdefault) );
  }


  int i=0;
  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  if(prefer == MIDDLE){
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       // prefer to stay on the left side then, but also check positions a bit more to the left
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
  }
  else if(prefer == RIGHT){
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
  }
  else{ // prefer LEFT
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
  }

  Tools::get_optimal_position(tp,testpos, i, teammate);

  //  optimize_position(tp,testpos, i, teammate);
  return tp;
}

Vector Attack_Move1_Nb::get_targetpos_centremidfielder() {
  Vector tp;
  float x_var = x_variation;
  float y_var = y_variation;
  float ydist2ball = 12.0;

  PPlayer teammate;
  WSinfo::get_teammate(6,teammate); // default teammate

  if(ball_is_left()) // then I relate my position to player 7
    WSinfo::get_teammate(6,teammate); 
  if(ball_is_right()) // then I relate my position to player 8
    WSinfo::get_teammate(8,teammate); 

  tp = Vector(offside_line - M_BEHIND_OFFSIDE_AGGRESSIVE,CM_Y);
  //  int prefer = MIDDLE;

  if(WSinfo::ball->pos.getY()>7){
    tp.setY( WSinfo::ball->pos.getY() - ydist2ball );
  }
  else if(WSinfo::ball->pos.getY()<-7){
    tp.setY( WSinfo::ball->pos.getY() + ydist2ball );
  }

#if 0
  if(Tools::is_pos_occupied_by_ballholder(tp)){
    DBLOG_POL(0,"My original targetpoint is occupied by ballholder ->adjust");
  }
#endif


  int i=0;
  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  testpos[i++] = Vector(tp.getX(),             tp.getY());
  testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
  testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
  testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
  testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);

  Tools::get_optimal_position(tp,testpos, i, teammate);

  //  optimize_position(tp,testpos, i, teammate);
  return tp;
}

Vector Attack_Move1_Nb::get_targetpos_rightmidfielder() {
  PPlayer teammate;
  WSinfo::get_teammate(7,teammate); // default teammate

  float xdist2teammate = 5.0;
  float ydist2teammate = 12.0;  // ridi: was 10.0 -> experiment with this

  float x_var = x_variation;
  float y_var = y_variation;


  // default position:
  Vector tp = Vector(offside_line - M_BEHIND_OFFSIDE_DEFAULT,RM_Y);
  int prefer = MIDDLE;

  if(WSinfo::ball->pos.getY() >0){ // ball is left
    if(teammate){
      tp = teammate->pos;
      tp.subFromX( xdist2teammate );
      tp.subFromY( ydist2teammate );
      tp.setY( Tools::min(tp.getY(),3.0) ); // do not go further than that
      tp.setX( Tools::max(offside_line - M_BEHIND_OFFSIDE_DEFAULT, tp.getX()) );
    }
    else{ // default, if 7 is not known
      tp.setY( 0.0 );
    }
    prefer=RIGHT;
  }
  else  if(WSinfo::ball->pos.getY() < RM_Y){
    tp.setY( WSinfo::ball->pos.getY() ); // if ball is left of my homepos, stay on ballheight
    prefer=LEFT;
  }

  int i=0;
  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  if(prefer == MIDDLE){
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       // prefer to stay on the left side then, but also check positions a bit more to the left
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
  }
  else if(prefer == RIGHT){
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
  }
  else{ // prefer LEFT
       testpos[i++] = Vector(tp.getX(),             tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX(),             tp.getY());
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY());
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY());
       testpos[i++] = Vector(tp.getX(),             tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + x_var,     tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX() + 2 * x_var, tp.getY() - y_var);
  }

  Tools::get_optimal_position(tp,testpos, i, teammate);

  return tp;
}

Vector Attack_Move1_Nb::get_targetpos_leftattacker() {
  float my_y = LA_Y;
  if(offside_line < 38)
    my_y = 15.0;

  Vector tp = Vector(offside_line,my_y);

  int i=0;
  float x_var = 3.0;
  float y_var = 3.0;

  PPlayer teammate;
  WSinfo::get_teammate(6,teammate); // default teammate
  if(teammate){
    if(teammate->pos.distance(WSinfo::ball->pos) > 10.){ // teammate is currently not involved
      teammate = NULL;
    }
  }

  tp.setY( Tools::max(WSinfo::ball->pos.getY() - 7, my_y) );

  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  testpos[i++] = Vector(tp.getX(),         tp.getY());
  testpos[i++] = Vector(tp.getX(),         tp.getY() - y_var);
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY());
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY() - y_var);
  testpos[i++] = Vector(tp.getX(),         tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY() + y_var);

  Tools::get_optimal_position(tp,testpos, i, teammate);
  return tp;
}

Vector Attack_Move1_Nb::get_targetpos_centreattacker() {
  Vector tp = Vector(offside_line,CA_Y);
  int i=0;

  PPlayer teammate;
  WSinfo::get_teammate(7,teammate); // default teammate


  float x_var = 3.0;
  float y_var = 3.0;

  int prefer = MIDDLE;
  if(Tools::num_teammates_in_circle(Vector(tp.getX(),tp.getY()+y_var),y_var) >
     Tools::num_teammates_in_circle(Vector(tp.getX(),tp.getY()-y_var),y_var)){
    DBLOG_POL(0,"more players to the left than to the right -> prefer right");
    prefer = RIGHT;
  }
  else 
    prefer = LEFT;

  if((Tools::num_teammates_in_circle(Vector(offside_line,5.0), 4.0) == 0) &&
     (WSinfo::ball->pos.getY() >0)){
    DBLOG_POL(0,"no left attacker and ball is left -> move tp to the left");
    tp.setY(  3.0 );
  }
  if((Tools::num_teammates_in_circle(Vector(offside_line,-5.0), 4.0) == 0) &&
     (WSinfo::ball->pos.getY() <0)){
    DBLOG_POL(0,"no right attacker and ball is right -> move tp to the right");
    tp.setY( -3.0 );
  }


  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  testpos[i++] = Vector(tp);
  if(prefer == RIGHT){
    testpos[i++] = Vector(tp.getX(),         tp.getY() - y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() - 2 * y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() + y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() + 2 * y_var);
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY());
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY() - y_var);
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY() + y_var);
  }
  else{ // prefer == LEFT
    testpos[i++] = Vector(tp.getX(),         tp.getY() - y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() - 2 * y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() + y_var);
    testpos[i++] = Vector(tp.getX(),         tp.getY() + 2 * y_var);
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY());
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY() - y_var);
    testpos[i++] = Vector(tp.getX() - x_var, tp.getY() + y_var);
  }
  Tools::get_optimal_position(tp,testpos, i, teammate);
  return tp;
}

Vector Attack_Move1_Nb::get_targetpos_rightattacker() {
  float my_y = RA_Y;
  if(offside_line < 38)
    my_y = -15.0;

  Vector tp = Vector(offside_line,my_y);
  int i=0;
  float x_var = 3.0;
  float y_var = 3.0;

  PPlayer teammate;
  WSinfo::get_teammate(8,teammate); // default teammate
  if(teammate){
    if(teammate->pos.distance(WSinfo::ball->pos) > 10.){ // teammate is currently not involved
      teammate = NULL;
    }
  }



  tp.setY( Tools::min(WSinfo::ball->pos.getY() + 7, my_y) );

  if(is_mypos_ok(tp)) // if it's possible, stay where you are.
    testpos[i++] = WSinfo::me->pos;  
  testpos[i++] = Vector(tp.getX(),         tp.getY());
  testpos[i++] = Vector(tp.getX(),         tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY());
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY() + y_var);
  testpos[i++] = Vector(tp.getX(),         tp.getY() - y_var);
  testpos[i++] = Vector(tp.getX() - x_var, tp.getY() - y_var);

  Tools::get_optimal_position(tp,testpos, i, teammate);
  return tp;
}

Vector Attack_Move1_Nb::get_targetpos() {
  Vector tp = Vector(0,0);
  switch(WSinfo::me->number){
  case 6:
    tp=get_targetpos_leftmidfielder();
    break;
  case 7:
    tp=get_targetpos_centremidfielder();
    break;
  case 8:
    tp=get_targetpos_rightmidfielder();
    break;
  case 9:
    tp=get_targetpos_leftattacker();
    break;
  case 10:
    tp=get_targetpos_centreattacker();
    break;
  case 11:
    tp=get_targetpos_rightattacker();
    break;

  }

  return tp;
}

bool Attack_Move1_Nb::do_waitandsee(Cmd &cmd){
    DBLOG_POL(0, << " DO WAIT AND SEE Face Ball");
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    return true;
}

bool Attack_Move1_Nb::go2pos_intelligent(Cmd &cmd, const Vector target){

  DBLOG_POL(0, << "go2pos intelligent: " << target);

  if((WSinfo::me->pos -target).norm()<10){
    // first check if there's someone between me and my target
    PlayerSet pset= WSinfo::valid_teammates_without_me;
    Vector endofregion = target - WSinfo::me->pos;
    if(endofregion.norm() > 3 * ServerOptions::kickable_area)
      endofregion.normalize(3 * ServerOptions::kickable_area);
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 4 * ServerOptions::kickable_area);
    //  DBLOG_DRAW(1, check_area );
    pset.keep_players_in(check_area);
    if(pset.num >0){
      DBLOG_POL(0,"There's a teammate on my way to my target, Wait and See ");
      return do_waitandsee(cmd);
    }
  }


  go2pos->set_target(target);
  if(go2pos->get_cmd(cmd) == false){ // no cmd was set; probably close enough
    return do_waitandsee(cmd);
  }
  return true;
}



bool Attack_Move1_Nb::test_offside(Cmd & cmd){

  if(WSinfo::me->pos.getX() > offside_line){
    Vector targetpos;
    targetpos.setXY( WSinfo::me->pos.getX() -6., WSinfo::me->pos.getY() );
    PlayerSet pset= WSinfo::valid_teammates_without_me;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, targetpos, 4.);
    //  DBLOG_DRAW(0,  C2D(endofregion.x, endofregion.y, 1., "red")); 
    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    pset.keep_players_in_circle(WSinfo::ball->pos, 2.0);
    if(pset.num >0){
      DBLOG_POL(0,"Disable offside: Ballholder within my target region -> modify target");
      targetpos.setX( WSinfo::me->pos.getX() );
      if(pset[0]->pos.getY() > targetpos.getY())
	targetpos.setY( WSinfo::me->pos.getY() - 10 );
      else
	targetpos.setY( WSinfo::me->pos.getY() + 10 );
    }

    DBLOG_POL(0, << "Disable Offside. My Pos"<<WSinfo::me->pos
	      <<" offence line "<< offside_line<<" mytarget: "<<targetpos);
    mdpInfo::set_my_intention(DECISION_TYPE_LEAVE_OFFSIDE,
			      (offside_line));
    // go one cycle to a position behind my current position
    go2pos->set_target( targetpos );
    go2pos->get_cmd(cmd);
    return true;
  }
  return false;
}


bool Attack_Move1_Nb::get_cmd(Cmd & cmd) {
  DBLOG_POL(0, << "ENTERED get_cmd");
  if(do_move() == false)
    return false;
  determine_offside_line();
  if(test_offside(cmd))
    return true;
  Vector targetpos = get_targetpos();
  DBLOG_POL(0, << "get_cmd");
  return go2pos_intelligent(cmd, targetpos);
}

bool Attack_Move1_Nb::ball_is_left() {
  if(WSinfo::ball->pos.getY() >0)
    return true;
  else
    return false;
}

bool Attack_Move1_Nb::ball_is_right() {
  if(WSinfo::ball->pos.getY() <0)
    return true;
  else
    return false;
}

bool Attack_Move1_Nb::ball_is_half_left() {
  if(WSinfo::ball->pos.getY() >10)
    return true;
  else
    return false;
}

bool Attack_Move1_Nb::ball_is_far_left() {
  if(WSinfo::ball->pos.getY() >20)
    return true;
  else
    return false;
}


