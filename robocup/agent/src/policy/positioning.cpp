#include "positioning.h"
#include "planning.h"
#include "mdp_info.h"
#include "tools.h"
#include "log_macros.h"
#include "options.h"
#include "valueparser.h"
#include "policy_tools.h"
#include "tools.h"
#include "ws_memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include "policy_goalie_kickoff.h"
//#include "strategy_defs.h"

//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)


#ifdef NEW433HACK
Formation433 DeltaPositioning::formation433;
Formation433Attack DeltaPositioning::attack433;
bool DeltaPositioning::use_new_formation433= false;
#endif

bool OpponentAwarePositioning::cvUseVersion2017 = true;

int           DeltaPositioning :: recommended_attentionto[NUM_PLAYERS+1];
DeltaPosition DeltaPositioning :: pos[NUM_CONSIDERED_POSITIONS];
bool          DeltaPositioning :: consider_player[11];
Formation     DeltaPositioning :: form[MAX_NUM_FORMATIONS];
int           DeltaPositioning :: cnt_formations;
int           DeltaPositioning :: current_formation;
int           DeltaPositioning :: current_matching;
int           DeltaPositioning :: current_evaluation;
double        DeltaPositioning :: ball_weight_x_defense;
double        DeltaPositioning :: ball_weight_y_defense;
double        DeltaPositioning :: ball_weight_x_buildup;
double        DeltaPositioning :: ball_weight_y_buildup;
double        DeltaPositioning :: ball_weight_x_attack;
double        DeltaPositioning :: ball_weight_y_attack;
double        DeltaPositioning :: stretch;
double        DeltaPositioning :: max_defense_line;
double        DeltaPositioning :: min_defense_line_offset;
double        DeltaPositioning :: defense_line_ball_offset;
//int           DeltaPositioning :: cycles_after_catch;
double        DeltaPositioning :: move_defense_line_to_penalty_area;
int           DeltaPositioning :: four_chain_established = 0;
bool          DeltaPositioning :: use_BS02_gamestates = 0;
double        DeltaPositioning :: before_goal_line = 28.0;

DashStamina   Stamina::dashstamina[3];
int           Stamina::stamina_management_type;
double        Stamina::stamina_reserve_defenders;
double        Stamina::stamina_reserve_midfielders;
double        Stamina::stamina_reserve_attackers;
double        Stamina::stamina_full_level;
int           Stamina::state;
int           Stamina::last_update;
double        Stamina::stamina_min_reserve_level;
double        Stamina::stamina_min_reserve;

const double NARROW_4_CHAIN_LINE = -30.0;

void DeltaPositioning :: init_formations(){
  //cout << "\nInit Formations...";
  current_formation = -1;
  cnt_formations = 0;
  move_defense_line_to_penalty_area = -20.0;
  char tchar[50];
  char key_pos[50];
  ValueParser vp(CommandLineOptions::formations_conf,"Definitions");
  //vp.set_verbose(true);
#ifdef NEW433HACK
  vp.get("use_new_formation433",use_new_formation433);
  formation433.init(CommandLineOptions::formations_conf,0,0);
  attack433.init(CommandLineOptions::formations_conf,0,0);
#endif
  vp.get("Initial_Formation", current_formation);
  vp.get("Initial_Matching" , current_matching);
  vp.get("Initial_Evaluation", current_evaluation);
  //vp.get("Ball_Weight_X", ball_weight_x);
  //vp.get("Ball_Weight_Y", ball_weight_y);
  vp.get("Ball_Weight_X_Defense", ball_weight_x_defense);
  vp.get("Ball_Weight_Y_Defense", ball_weight_y_defense);
  vp.get("Ball_Weight_X_Buildup", ball_weight_x_buildup);
  vp.get("Ball_Weight_Y_Buildup", ball_weight_y_buildup);
  vp.get("Ball_Weight_X_Attack", ball_weight_x_attack);
  vp.get("Ball_Weight_Y_Attack", ball_weight_y_attack);
  
  vp.get("Stretch", stretch);
  vp.get("min_defense_line_offset",min_defense_line_offset);
  vp.get("max_defense_line",max_defense_line);
  vp.get("defense_line_ball_offset",defense_line_ball_offset);
  vp.get("move_defense_line_to_penalty_area",move_defense_line_to_penalty_area);
  vp.get("use_BS02_gamestates", use_BS02_gamestates);
  vp.get("before_goal_line", before_goal_line);

//  cout << " use_BS02_gamestates = " << use_BS02_gamestates;
//  cout << " before_goal_line = " << before_goal_line;

  sprintf(key_pos, "%d", cnt_formations);
  while (vp.get(key_pos, tchar, 50)>0){
    form[cnt_formations].number = cnt_formations;
    strncpy(form[cnt_formations].name, tchar, 50);
    ValueParser vp2(CommandLineOptions::formations_conf,tchar);
    //vp2.set_verbose(true);
    for(int i=1;i<=10;i++){
      sprintf(key_pos, "pos_%d",i);
      vp2.get(key_pos, form[cnt_formations].pos[i-1], 3);
    }
    cnt_formations++;
    sprintf(key_pos, "%d", cnt_formations);
  }
  cnt_formations--;
  if (cnt_formations == -1){
    cout << "\nNo Formations found.";
  }
  /****************************STAMINA*********************/
  Stamina::init();
#if 0
  cout<<" Positioning: Initial Formation" << current_formation
      <<"Initial Matching" << current_matching
      <<"Initial Evaluation"<< current_evaluation
      <<"Ball Weight X"<< ball_weight_x
      <<"Ball Weight Y" <<ball_weight_y
      <<"Stretch"<<stretch
      <<"min_defense_line_offset"<<min_defense_line_offset
      <<"max_defense_line"<<max_defense_line
      <<"defense_line_ball_offset"<<defense_line_ball_offset;
#endif

  //ValueParser vp3(CommandLineOptions::formations_conf,"Attention");
                        
  ValueParser vp3("../conf/attention.conf","Attention");
  //vp3.set_verbose(true);
  char key_player[40];
  for(int i=1; i < NUM_PLAYERS + 1; i++) {
    sprintf(key_player, "player_%d",i);
    recommended_attentionto[i]= -1;
    vp3.get(key_player, recommended_attentionto[i]);
  }
}

DeltaPositioning :: DeltaPositioning(){
  for(int i=0; i<11;i++){
    consider_player[i] = true;
  }
}

void DeltaPositioning :: init_players(bool *players){
  for(int i=0; i<11;i++){
    consider_player[i] = players[i];
  }
}

DashPosition DeltaPositioning::get_position(int player_number){
  DashPosition temp;
  if ( use_new_formation433 ) {
    Vector res= formation433.get_fine_pos(player_number);
    temp.clone( res );
  }
  return temp; 
}

Vector DeltaPositioning::get_position_base(int player_number){
  if ( use_new_formation433 ) 
    return formation433.get_grid_pos(player_number);
  ERROR_OUT << "\nshould never reach this point";
  return Vector(0,0); 
}

double DeltaPositioning :: get_my_defence_line() {
  if ( use_new_formation433 ) {
    double defence_line, offence_line;
    formation433.get_boundary(defence_line, offence_line);
    return defence_line;
  }
 ERROR_OUT << "\nshould never reach this point (DeltaPositioning::get_my_defence_line()";
 return 0.0;  
  //Daniel return get_my_defence_line_BS02();
}


double DeltaPositioning :: get_my_offence_line(){
  if ( use_new_formation433 ) {
    double defence_line, offence_line;
    formation433.get_boundary(defence_line, offence_line);
    return offence_line;
  }
  ERROR_OUT << "\nshould never reach this point (DeltaPositioning::get_my_offence_line()";
  return 0.0;  
}

int DeltaPositioning :: get_role(int player_number){
  if ( use_new_formation433 ) {
    return formation433.get_role(player_number);
  }
  ERROR_OUT << "\nshould never reach this point (DeltaPositioning::get_role()";
  return 0;
}

/** 
    #####################################################
    STAMINA MANAGEMENT
    #####################################################
 */


double Stamina::economy_level(){
  int role = DeltaPositioning::get_role(mdpInfo::mdp->me->number);
  double min_level = ServerOptions::recover_dec_thr*ServerOptions::stamina_max;

  if(role==0) // defender
    return(min_level + stamina_reserve_defenders);
  else if(role==1) // midfielder
    return(min_level + stamina_reserve_midfielders);
  else // attacker
    return(min_level + stamina_reserve_attackers);
}

void Stamina::update_state(){
#define THRESHHOLD 200

  if(mdpInfo::mdp->time_current == last_update)
    return;
  double stamina = mdpInfo::mdp->me->stamina.v;

  switch(state){
  case STAMINA_STATE_FULL:
    if(stamina<stamina_full_level)
      state = STAMINA_STATE_OK;
    break;
  case STAMINA_STATE_OK:
    if(stamina>stamina_full_level + THRESHHOLD)
      state = STAMINA_STATE_FULL;
    else if(stamina<economy_level())
      state= STAMINA_STATE_ECONOMY;
    break;
  case STAMINA_STATE_ECONOMY:
    if(stamina>economy_level()+THRESHHOLD)
      state=STAMINA_STATE_OK;
    else if(stamina<stamina_min_reserve_level)
      state=STAMINA_STATE_RESERVE;
    break;
  case STAMINA_STATE_RESERVE:
    if (stamina>stamina_min_reserve_level+THRESHHOLD)
      state=STAMINA_STATE_ECONOMY;
    break;
  default:
    if(stamina>stamina_full_level)
      state=STAMINA_STATE_FULL;
    else if(stamina>economy_level())
      state=STAMINA_STATE_OK;
    else if(stamina>stamina_min_reserve_level)
      state=STAMINA_STATE_ECONOMY;
    else
      state=STAMINA_STATE_RESERVE;
    break;
  }
  last_update = mdpInfo::mdp->time_current;
}

int Stamina::get_state(){
  update_state();
  return state;
}


int Stamina::dash_power(){
  int role = DeltaPositioning::get_role(mdpInfo::mdp->me->number);
  double stamina = WSinfo::me->stamina;

  if(stamina_management_type == 0){ // using default Stamina Management
    if(role==0){
      //Verteidiger orientieren sich an Ball
      if(mdpInfo::is_my_team_attacking() && WSinfo::ball->pos.getX() > WSinfo::me->pos.getX()){
	if(stamina >= dashstamina[role].stamina_offence) return (int)dashstamina[role].dash_offence;
      }
      if(mdpInfo::is_my_team_attacking() && WSinfo::ball->pos.getX() <= WSinfo::me->pos.getX()){
	if(stamina >= dashstamina[role].stamina_defence) return (int)dashstamina[role].dash_defence;
      }
    } 
    
    if(mdpInfo::is_my_team_attacking()){
      if(stamina >= dashstamina[role].stamina_offence) return (int)dashstamina[role].dash_offence;
    } else{
      if(stamina >= dashstamina[role].stamina_defence) return (int)dashstamina[role].dash_defence;
    }
    return 0;
  }
  else{  // using new stamina management
    if(get_state() != STAMINA_STATE_RESERVE){
      LOG_POL(4,"Stamina Management: My stamina "<<stamina<<" is still ok "
	      <<" -> Dashing with Maximum Power: "<<ServerOptions::maxpower);
      return (int)ServerOptions::maxpower;
    }
    else{ // I'm close to end with my stamina, keeping iron reserve
      LOG_POL(4,"Stamina Management: My stamina "<<stamina<<" in Reserve State! "
	      <<" -> Dashing with Minimum Power: "<<ServerOptions::stamina_inc_max);
      return (int)ServerOptions::stamina_inc_max;
    }

#if 0 // ridi: that was wrong, because it does not distinguish between urgent situations    
    float stamina_reserve;
    if(role==0) // defender
      stamina_reserve = stamina_reserve_defenders;
    else if(role==1) // midfielder
      stamina_reserve = stamina_reserve_midfielders;
    else // attacker
      stamina_reserve = stamina_reserve_attackers;
    if(stamina > stamina_reserve + ServerOptions::recover_dec_thr*ServerOptions::stamina_max){
      LOG_POL(4,"Stamina Management: My stamina "<<stamina<<" is larger than min.reserve "
	      <<stamina_reserve+ ServerOptions::recover_dec_thr*ServerOptions::stamina_max
	      <<" -> Dashing with Maximum Power: "<<ServerOptions::maxpower);
      return (int)ServerOptions::maxpower;
    }
    else{    
      LOG_POL(4,"Stamina Management: My stamina "<<stamina<<" is smaller than min.reserve "
	      <<stamina_reserve+ ServerOptions::recover_dec_thr*ServerOptions::stamina_max
	      <<" -> Dashing with Minimum Power: "<<ServerOptions::stamina_inc_max);
      return (int)ServerOptions::stamina_inc_max;
    }
#endif
  } // end new management
}


void Stamina::init(){
  double temp[4];
  //cout << "\nInit Stamina...";
  stamina_reserve_defenders = 1000;
  stamina_reserve_midfielders = 500;
  stamina_reserve_attackers = 500;
  stamina_min_reserve = 200;
  stamina_full_level = ServerOptions::stamina_max * 0.8; // all above this is considered as full

  ValueParser vp(CommandLineOptions::formations_conf,"Stamina");
  //vp.set_verbose(true);
  stamina_management_type = 0; // this is the default: Andis (BS2k) Stamina Management
  vp.get("stamina_for_defence", temp,4);
  dashstamina[0].stamina_defence = temp[0];
  dashstamina[0].dash_defence    = temp[1];
  dashstamina[0].stamina_offence = temp[2];
  dashstamina[0].dash_offence    = temp[3];
  vp.get("stamina_for_middle", temp,4);
  dashstamina[1].stamina_defence = temp[0];
  dashstamina[1].dash_defence    = temp[1];
  dashstamina[1].stamina_offence = temp[2];
  dashstamina[1].dash_offence    = temp[3];
  vp.get("stamina_for_offence", temp,4);
  dashstamina[2].stamina_defence = temp[0];
  dashstamina[2].dash_defence    = temp[1];
  dashstamina[2].stamina_offence = temp[2];
  dashstamina[2].dash_offence    = temp[3];
  vp.get("stamina_management_type", stamina_management_type);
  vp.get("stamina_reserve_defenders", stamina_reserve_defenders);
  vp.get("stamina_reserve_midfielders", stamina_reserve_midfielders);
  vp.get("stamina_reserve_attackers", stamina_reserve_attackers);
  stamina_min_reserve_level = stamina_min_reserve +
    ServerOptions::recover_dec_thr*ServerOptions::stamina_max;
  if(stamina_management_type == 0){
    //LOG_ERR(0,<<"Initialized Stamina Management. using original stamina management (BS2k)");
//    cout<<"Initialized Stamina Management. using original stamina management (BS2k)"<<endl;
  }
  else if(stamina_management_type == 1){
    //LOG_ERR(0,<<"Initialized Stamina Management. using new stamina management (BS01)");
//    cout<<"Initialized Stamina Management. using new stamina management (BS01)"<<endl;
  }
  else
    stamina_management_type = 0; // default
  state = STAMINA_STATE_FULL;
  last_update = 0;
}

/*
void DeltaPositioning :: match_positions_by_moveup(){
  //gegeben: Position der Spieler von mdpInfo + Position der Formation
  //gesucht: matching zu jeder Position

  bool conflict = true;
  int unmatched[10]; //haelt Menge der ungematchten Positionen
  int match[10];     //haelt das Matching fuer Spieler auf Positionen
  int unmatch_cnt = 0; //haelt Anzahl der ungematchten Positionen

  for(int i=0;i<10;i++){
    match[i] = -1;
  }

  do{

    //0. Match unmatched Position
    if(unmatch_cnt>0){
      //bestime schnellsten Spieler zu unmatched[0]
      double min=1000.0;
      double temp;
      int fastest_player = -1;
      for(int i=0;i<10;i++){
        if(match[i] >= 0) continue;
        temp = evaluate(i+2, unmatched[0]);
        if(temp<min){
          min=temp;
          fastest_player = i;
	}
      }
      if(fastest_player >= 0) 
        match[fastest_player] = unmatched[0];
      else
        break;
    }

    //1. Greedy Matching with remaining players
    int tpos[10];
    double min, temp;
    for(int i=0;i<10;i++){
      tpos[i] = -1;
      if(match[i] >= 0) continue;
      min=10000.0;
      for(int j=0;j<NUM_CONSIDERED_POSITIONS;j++){
        temp = evaluate(i+2, j);
        if (temp<min){
          min = temp;
          tpos[i] = j;
        }
      }
    }
    
    //2. Still Conflicts?

    conflict=false;
    for(int i=0;i<10;i++){
      if(tpos[i]<0) continue;
      for(int j=i+1;j<10;j++){
        if(tpos[j]<0) continue;
        if(tpos[i] == tpos[j])  conflict = true;
      }
    }
    //Bestimme Menge der ungematchten Positionen
    if(conflict==true){
      unmatch_cnt = 0;
      bool matched = false;
      for(int i=0;i<10;i++){
        matched = false;
        //1. Temp-Matching
        for(int j=0;j<10;j++){
          if(tpos[j] == i) matched = true;
        }
	//2. End-Matching
        for(int j=0;j<10;j++){
          if(match[j] == i) matched = true;
	}
        if(matched==false){
          unmatched[unmatch_cnt]=i;
          unmatch_cnt++;
	}
      }
    } else{
      //?bernehme Temp-Matching in End-Matching
      for(int i=0;i<10;i++){
        if(tpos[i]<0) continue;
        match[i] = tpos[i];
      }
    }
  } while(conflict == true);
  // da Matching nicht stabil, pruefe mit vorigem Matching ab
  // bestimme maximalen Wegstrecke fuer neues Matching
  double max = 0;
  double temp;
  for(int i=0;i<10;i++){
    temp = evaluate(i+2, match[i]);
    if(max < temp) max = temp;
  }
  //bestimme maximale Wegstrecke fuer altes Matching
  double max2 = 0;
  for(int i=0;i<10;i++){
    temp = evaluate(pos[i].player_assigned, i);
    if(max2 < temp) max2 = temp;
  }  
  //wenn altes Matching besser, nehme altes!
  if(max2 < max) return;  

  //schreiben auf wahre variablen
  for(int i=0;i<10;i++){
    if(match[i]<0) continue;
    pos[match[i]].player_assigned = i+2; 
    LOG_POL(4, << "Player " << i+2 << " soll auf Position " << match[i] << " " << pos[match[i]] << "gehen");
  }
}
*/


double DeltaPositioning::get_radius(int player_number){
  //stelle fest, ob ich im Radius um meine Zielposition bin
  double radius = 1000;

  int posid=-1;
  Vector p = Vector(0.0,0.0);
  for(int i=0; i<NUM_CONSIDERED_POSITIONS;i++){
    if (pos[i].player_assigned == mdpInfo::mdp->me->number){
      posid = i;
      p.clone( pos[i] );
    }
  }
  if(posid==-1) return 0.0;
  
  //ich wurde gematcht
  for(int i=0;i<NUM_CONSIDERED_POSITIONS;i++){
    if(i==posid) continue;
    if(p.distance(pos[i])<radius){
      radius = p.distance(pos[i]);
    }
  }
  if(radius == 1000) return 0.0;
  radius = radius / 2.0;
  return radius;
}

bool DeltaPositioning::is_position_valid(Vector p){
  //Abseits?
  if(p.getX() > get_my_offence_line()) return false;
  
  //ausserhalb des Spielfeldes?
  if(p.getX() >  FIELD_BORDER_X ||
     p.getX() < -FIELD_BORDER_X ||
     p.getY() >  FIELD_BORDER_Y ||
     p.getY() < -FIELD_BORDER_Y) return false;

  //ausserhalb meines zugewiesenen Zielbereiches?
  DashPosition default_pos = get_position(mdpInfo::mdp->me->number);
  if(default_pos.distance(p) >= default_pos.radius) return false;
  return true;
}

int DeltaPositioning::get_num_players_with_role(int role){
  int num = 0;
  for(int i=0; i<NUM_CONSIDERED_POSITIONS;i++){
    if((int)form[current_formation].pos[i][2]==role) num++;
  }
  return num;
}

double DeltaPositioning::get_defence_line_ball_offset() {
#ifdef NEW433HACK
  if ( use_new_formation433 ) {
    return formation433.defence_line_ball_offset;
  }
#endif
  return defense_line_ball_offset;
}

//////////////////////////////////////////////////////////////////////////////
// EXTENSIONS by TGA 2005
//////////////////////////////////////////////////////////////////////////////
#if 1
#define   TGLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGLOGPOL(YYY,XXX)
#endif

//STATIC VARIABLE
Formation05           OpponentAwarePositioning::cvFormation;
tInterceptInformation OpponentAwarePositioning::cvInterceptInformation;

//============================================================================
// INIT
//============================================================================
/**
 * init method of OpponentAwarePositioning, mainly invokes init of formations.
 */
bool 
OpponentAwarePositioning::
  init(char const * confFile, int argc, char const* const* argv) 
{
  return initFormations();
}

//============================================================================
// initFormations
//============================================================================
/**
 * initFormations invokes the initialization of all formations used.
 * 
 */
bool
OpponentAwarePositioning::initFormations()
{
  return cvFormation.init(CommandLineOptions::formations_conf, 0, 0);
}

//============================================================================
// getRole
//============================================================================
/**
 * gets the role of the player indicated by number 
 */
int
OpponentAwarePositioning::getRole(int number)
{
  return cvFormation.get_role(number);
}

//============================================================================
// getHomePosition
//============================================================================
/**
 * gets my home position in a rectangle
 */
OpponentAwarePosition 
OpponentAwarePositioning::getHomePosition(XYRectangle2d r)
{
  OpponentAwarePosition targetPosition;
  targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number,r) );
  return targetPosition;
}
//============================================================================
// getHomePosition
//============================================================================
/**
 * gets my home position (home positions are almost not used for Formation05)
 */
OpponentAwarePosition 
OpponentAwarePositioning::getHomePosition()
{
  OpponentAwarePosition targetPosition;
  targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  return targetPosition;
}

//============================================================================
// getCoverPosition
//============================================================================
/**
 * getCoverPosition is a crucial method for our defense and for 
 * OpponentAwarePositioning in general. It calculates the best cover
 * position to which I should move in order to cover my direct opponent
 * optimally.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getCoverPosition()
{
  OpponentAwarePosition targetPosition;
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    targetPosition = getCoverPosition( directOpponent );
  }
  else
  {
    //according to my formation, i currently have no direct opponent
    TGLOGPOL(0,<<"NO OPPONENT FOUND => go to home position"<<flush);
    targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  }
  return targetPosition;
}

//============================================================================
// getAngleToDirectOpponent
//============================================================================
/**
 */
bool
OpponentAwarePositioning::getAngleToDirectOpponent( ANGLE & angleToDirOpp)
{
  PPlayer directOpponent;
  angleToDirOpp = ANGLE(0.0);
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    angleToDirOpp = (directOpponent->pos - WSinfo::me->pos).ARG();
    if (directOpponent->age > 1)
      return false;
    return true;
  }
  else
    return false;
}

//============================================================================
// getStaminaSavingCoverPosition
//============================================================================
/**
 * Computes a cover position for me taking into consideration my stamina.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getStaminaSavingCoverPosition()
{
  OpponentAwarePosition targetPosition;
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    targetPosition = getStaminaSavingCoverPosition( directOpponent );
  }
  else
  {
    //according to my formation, i currently have no direct opponent
    TGLOGPOL(0,<<"NO OPPONENT FOUND => go to home position"<<flush);
    targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  }
  return targetPosition;
}

//============================================================================
// getStaminaSavingCoverPosition
//============================================================================
/**
 * Computes a cover position for the specified player
 * taking into consideration his stamina.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getStaminaSavingCoverPosition( PPlayer directOpponent )
{
  //start the computation of an appropriate cover position
  OpponentAwarePosition targetPosition;
  //begin of computation
    targetPosition.opponentNumber = directOpponent->number;
    //1. basierend auf aktueller opponentPosition berechne die voraussichtliche
    //   position im n?chsten zyklus ==> alpha
    //   annahmen: opponent dasht mit 40 in seine aktuelle richtung
       //position of my direct opponent
      const Vector currentOpponentPosition = directOpponent->pos; 
      Vector tempOpponentVelocity = directOpponent->vel;
      if ( directOpponent->age_vel > 3 )
      { //keine information ueber die geschwindigkeit des gegners -> schaetzung
        tempOpponentVelocity.setX( -0.4 ); //annahme: leichte vorwaertsbewegung
        //leicht bewegung in richtung unseres tores
        tempOpponentVelocity.setY(
            - (currentOpponentPosition.getY() / FIELD_BORDER_Y) * 0.5 );
      }
      const Vector currentOpponentVelocity = tempOpponentVelocity;
      Angle tempOpponentAngle = directOpponent->ang.get_value();
      if ( directOpponent->age_ang > 1 )
      {
        Angle oppAng2Goal = currentOpponentPosition.angle( MY_GOAL_CENTER );
        Angle oppAng2Ball = currentOpponentPosition.angle( WSinfo::ball->pos );
        tempOpponentAngle = (oppAng2Goal+oppAng2Ball) / 2.0;
      }
      const Angle currentOpponentAngle = tempOpponentAngle;
      Cmd_Body assumedOpponentCommand;
      assumedOpponentCommand.set_dash(40);
      
      Vector assumedNextOpponentPosition, assumedNextOpponentVelocity;
      Angle assumedNextOpponentAngle;
      int dummyStamina;
      Vector newBallPos, newBallVel;
      Tools::setModelledPlayer(directOpponent);
      Tools::model_cmd_main( currentOpponentPosition,
                             currentOpponentVelocity,
                             currentOpponentAngle,
                             4000,
                             WSinfo::ball->pos,
                             WSinfo::ball->vel,
                             assumedOpponentCommand,
                             assumedNextOpponentPosition,
                             assumedNextOpponentVelocity,
                             assumedNextOpponentAngle,
                             dummyStamina,
                             newBallPos,
                             newBallVel,
                             false //do_random==false
                           );
     Tools::setModelledPlayer(WSinfo::me);
     TGLOGPOL(1,<<_2D<<VC2D(assumedNextOpponentPosition,0.3,"4444ff")<<flush);
  
    //2. berechne lotfuss von mir zur strecke "alpha bis eigenes tores"
    Vector goalReferencePoint;
    goalReferencePoint.setX( -FIELD_BORDER_X );
    goalReferencePoint.setY( directOpponent->pos.getY() );
    if (directOpponent->pos.getY() > 7.0) goalReferencePoint = MY_GOAL_LEFT_CORNER;
    if (directOpponent->pos.getY() <-7.0) goalReferencePoint = MY_GOAL_RIGHT_CORNER;
    Vector lotfuss;
    lotfuss = Tools::get_Lotfuss( goalReferencePoint, 
                                  directOpponent->pos, 
                                  WSinfo::me->pos );
TGLOGPOL(0,<<"LOTFUSSINFO: x1="<<goalReferencePoint.getX()<<","<<goalReferencePoint.getY()<<"   x2="<<directOpponent->pos.getX()<<","<<directOpponent->pos.getY()<<"   p="<<WSinfo::me->pos.getX()<<","<<WSinfo::me->pos.getY()<<"   ==> lf="<<lotfuss.getX()<<","<<lotfuss.getY());
    //3. ergebnis speichern
    targetPosition.clone( lotfuss );
    //4. ergebnis verifizieren: liegt die berechnete target position vor
    //dem direkten gegenspieler, so sollte lieber die standarddeckposition
    //zurueckgegeben werden
    if ( targetPosition.getX() > directOpponent->pos.getX() )
      return getCoverPosition( directOpponent );
    
  //end of computation
  return targetPosition;
}

//============================================================================
// getCoverPosition
//============================================================================
/**
 * This version of getCoverPosition provides the "standard" interface
 * to this method.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getCoverPosition( PPlayer directOpponent )
{
  return 
    getCoverPosition( directOpponent->number,
                      directOpponent->pos,
                      directOpponent->vel,
                      directOpponent->ang,
                      directOpponent->age,
                      directOpponent->age_vel,
                      directOpponent->age_ang );
}

//============================================================================
// getCoverPosition
//============================================================================
/**
 * getCoverPosition computes the optimal cover position to cover the
 * specified opponent player (directOpponent).
 */
OpponentAwarePosition 
OpponentAwarePositioning::getCoverPosition //( PPlayer directOpponent )
                          (  const int    & directOpponentNumber,
                             const Vector & directOpponentPos,
                             const Vector & directOpponentVel,
                             const ANGLE  & directOpponentANG,
                             const int    & directOpponentAge,
                             const int    & directOpponentAgeVel,
                             const int    & directOpponentAgeAng
                          )
{
  PPlayer directOpponent 
    = WSinfo::alive_teammates.get_player_by_number( directOpponentNumber );
  //start the computation of an appropriate cover position
  OpponentAwarePosition targetPosition;
  //begin of computation
    targetPosition.opponentNumber = directOpponentNumber;
    //1. basierend auf aktueller opponentPosition berechne die voraussichtliche
    //   position im n?chsten zyklus ==> alpha
    //   annahmen: opponent dasht mit 70 in seine aktuelle richtung
       //position of my direct opponent
      const Vector currentOpponentPosition = directOpponentPos; 
      Vector tempOpponentVelocity = directOpponentVel;
      if ( directOpponentAgeVel > 3 )
      { //keine information ueber die geschwindigkeit des gegners -> schaetzung
        tempOpponentVelocity.setX( -0.6 ); //annahme: leichte vorwaertsbewegung
        //leicht bewegung in richtung unseres tores
        tempOpponentVelocity.setY(
            - (currentOpponentPosition.getY() / FIELD_BORDER_Y) * 0.5 );
      }
      const Vector currentOpponentVelocity = tempOpponentVelocity;
      Angle tempOpponentAngle = directOpponentANG.get_value();
      if ( directOpponentAgeAng > 1 )
      {
        Angle oppAng2Goal = currentOpponentPosition.angle( MY_GOAL_CENTER );
        Angle oppAng2Ball = currentOpponentPosition.angle( WSinfo::ball->pos );
        tempOpponentAngle = (oppAng2Goal+oppAng2Ball) / 2.0;
      }
      const Angle currentOpponentAngle = tempOpponentAngle;
      Cmd_Body assumedOpponentCommand;
      assumedOpponentCommand.set_dash(80);
      
      Vector assumedNextOpponentPosition, assumedNextOpponentVelocity;
      Angle assumedNextOpponentAngle;
      int dummyStamina;
      Vector newBallPos, newBallVel;
      if (directOpponent) Tools::setModelledPlayer(directOpponent);
      Tools::model_cmd_main( currentOpponentPosition,
                             currentOpponentVelocity,
                             currentOpponentAngle,
                             4000,
                             WSinfo::ball->pos,
                             WSinfo::ball->vel,
                             assumedOpponentCommand,
                             assumedNextOpponentPosition,
                             assumedNextOpponentVelocity,
                             assumedNextOpponentAngle,
                             dummyStamina,
                             newBallPos,
                             newBallVel,
                             false //do_random==false
                           );
      Tools::setModelledPlayer(WSinfo::me);
     TGLOGPOL(1,<<_2D<<VC2D(assumedNextOpponentPosition,0.3,"4444ff")<<flush);
    
    //2. berechne punkt auf der strecke "alpha bis mitte eigenen tores", der
    //   20% von alpha entfernt ist ==> beta
    Vector goalReferencePoint = MY_GOAL_CENTER;
    if ( assumedNextOpponentPosition.getX() < -0.65*FIELD_BORDER_X )
    {
      float yOffset = 0.0;
      if ( fabs(assumedNextOpponentPosition.getY()) <= 7.0)
        yOffset = assumedNextOpponentPosition.getY();
      if ( assumedNextOpponentPosition.getY() > 7.0 ) yOffset = 7.0;
      if ( assumedNextOpponentPosition.getY() < -7.0 ) yOffset = -7.0;
      goalReferencePoint = Vector(-FIELD_BORDER_X, yOffset);
    }
    double dichte=0.8;
    if (   WSinfo::ball->pos.getX() > FIELD_BORDER_X - 20.0
        || OpponentAwarePositioning::getRole(WSinfo::me->number)==PT_MIDFIELD
        || OpponentAwarePositioning::getRole(WSinfo::me->number)==PT_ATTACKER
        || WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 20.0
       )
    {
      dichte = 0.95;
      TGLOGPOL(0,<<"DICHT CHANGE I: "<<dichte);
    }
    else
    {
      TGLOGPOL(0,<<"DICHT CHANGE I: NOPE ballX:"<<WSinfo::ball->pos.getX()<<" role:"
        <<OpponentAwarePositioning::getRole(WSinfo::me->number)<<" ball2ol:"
        <<WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line());
    }
    if (   directOpponent
        && directOpponent->pos.getX() < WSinfo::me->pos.getX()
        && WSmemory::cvCurrentInterceptPeople.num > 0
        && WSmemory::cvCurrentInterceptPeople[0]->number == directOpponent->number
       )
    {
      dichte = 0.8;
      TGLOGPOL(0,<<"DICHT CHANGE II: "<<dichte);
    }
if (//  (  OpponentAwarePositioning::getRole(WSinfo::me->number==PT_DEFENDER)
    //   && WSinfo::ball->pos.x - WSinfo::my_team_pos_of_offside_line() > 15.0
    //   && WSinfo::me->pos.x - WSinfo::my_team_pos_of_offside_line() > 10.0
    //   && WSinfo::ball->pos.distance(MY_GOAL_CENTER) > 20.0 )
    //||
      (  OpponentAwarePositioning::getRole(WSinfo::me->number)==PT_MIDFIELD
//       && WSinfo::me->pos.x > -FIELD_BORDER_X * 0.1
  //     && WSinfo::me->pos.x - WSinfo::my_team_pos_of_offside_line() > 10.0
    //   && WSinfo::ball->pos.x - WSinfo::my_team_pos_of_offside_line() > 10.0
      )
   )
{
  if (   WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_WRIGHTEAGLE
      || WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_OXSY
      || WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_CYRUS )
  {
    dichte = 0.95;
    TGLOGPOL(0,"DICHTE-INCREMENT-0.95!");
    TGLOGPOL(0,"DICHTE CHANGE III: "<<dichte);
  }
}
    if (    WSinfo::ws->time - WSmemory::last_kick_off_at < 50
         || fabs( (assumedNextOpponentPosition-MY_GOAL_CENTER)//gegner auf aussen!
                  .ARG().get_value_mPI_pPI() ) > PI*(75./180.) )
{
  if (WSinfo::ws->time >= 50) {  TGLOGPOL(0,"DICHTE-REDUCTION-0.7!");}
  {
      dichte = 0.7;
      TGLOGPOL(0,"DICHTE CHANGE IV: "<<dichte);
  }
}

    if (   WSinfo::ws //TG17 FINAL HELIOS HACK
        && WSinfo::ws->his_team_score <= WSinfo::ws->my_team_score
        && WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS
        && !(   assumedNextOpponentPosition.getX() < -27.0
             || assumedNextOpponentPosition.distance(MY_GOAL_CENTER) < 23.0
             || WSinfo::ball->pos.getX() < -27.0
            )
       )
    {
      dichte = 0.675;
    }

    Vector firstCoverPoint = (dichte/*ZUI:0.8*/ * (assumedNextOpponentPosition 
                                    - goalReferencePoint)) + goalReferencePoint;

    double yFloatFactor = Tools::min(0.95 - dichte, 0.05);
    firstCoverPoint.setX( (dichte * (assumedNextOpponentPosition.getX()
                                    - goalReferencePoint.getX())) + goalReferencePoint.getX() );
    double dichteY = ( fabs(assumedNextOpponentPosition.getY())/FIELD_BORDER_Y ) * yFloatFactor + dichte;
    firstCoverPoint.setY( (dichteY * (assumedNextOpponentPosition.getY()
                                    - goalReferencePoint.getY())) + goalReferencePoint.getY() );


    TGLOGPOL(1,<<_2D<<VC2D(firstCoverPoint,0.2,"4444ff")<<flush);
    
    //3. berechne punkt auf der strecke "alpha bis aktuelle ballposition", der
    //   10% von alpha entfernt ist ==> gamma
    //   sonderregelung: je nach erwarteter gegnerposition im naechsten zyklus
    //   (alpha) wird dieser prozentwert sukzessive auf 0% reduziert
    float ballIgnoranceThreshold_MinOppDist2Goal = 15.0;
    float ballIgnoranceThreshold_MaxOppDist2Goal = 25.0;
    float assumedOpponentDistToGoal = (assumedNextOpponentPosition
                                       - MY_GOAL_CENTER).norm();
    float ballDirectionConsideration;
    if (assumedOpponentDistToGoal > ballIgnoranceThreshold_MaxOppDist2Goal)
      ballDirectionConsideration = 0.9;
    else
      if (assumedOpponentDistToGoal < ballIgnoranceThreshold_MinOppDist2Goal)
        ballDirectionConsideration = 1.0;
      else
      {
        float offset = 0.1 *
          ((assumedOpponentDistToGoal - ballIgnoranceThreshold_MinOppDist2Goal)
           / (ballIgnoranceThreshold_MaxOppDist2Goal - ballIgnoranceThreshold_MinOppDist2Goal));
        ballDirectionConsideration = 1.0 - offset;
      }  
    Vector secondCoverPoint = ballDirectionConsideration * (assumedNextOpponentPosition
                                                           - newBallPos ) + newBallPos;
    TGLOGPOL(1,<<_2D<<VC2D(newBallPos,0.3,"ff9900")<<flush);
    TGLOGPOL(1,<<_2D<<VL2D(firstCoverPoint,secondCoverPoint,"magenta")<<flush);
    TGLOGPOL(1,<<_2D<<VC2D(secondCoverPoint,0.2,"ff9900")<<flush);
    
    //sonderfall (gilt nicht fuer abwehr): wenn ball sehr nahe am gegenspieler,
    //ist ein naeherer deckpunkt wuenschenswert
    PlayerSet playersNearCovPos = WSinfo::valid_teammates_without_me;
    playersNearCovPos.append( WSinfo::valid_opponents );
    playersNearCovPos.remove( directOpponent );
    Vector targetPositionAsVector = 0.5 * (firstCoverPoint + secondCoverPoint);
    playersNearCovPos.keep_players_in_circle(targetPositionAsVector,
      targetPositionAsVector.distance(secondCoverPoint) * 0.5 );
    if (1 && WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 20.0
          && (   WSinfo::ball->pos.distance(assumedNextOpponentPosition) < 10.0
              || playersNearCovPos.num > 1))
    {
      OpponentAwarePosition changedTargetPosition;
      changedTargetPosition.setX(   0.25 *  firstCoverPoint.getX() );
      changedTargetPosition.addToX( 0.75 * secondCoverPoint.getX() );
      changedTargetPosition.setY(   0.25 *  firstCoverPoint.getY() );
      changedTargetPosition.addToY( 0.75 * secondCoverPoint.getY() );
      TGLOGPOL(1,<<_2D<<VC2D(changedTargetPosition,0.2,"magenta")<<flush);
      TGLOGPOL(1,<<_2D<<VC2D(changedTargetPosition,0.3,"magenta")<<flush);
      return changedTargetPosition;
    }
    
    //4. berechne der mittelpunkt der strecke "beta bis gamma" ==> targetPos
    Vector coverPoint = 0.5 * (firstCoverPoint + secondCoverPoint);
     TGLOGPOL(1,<<_2D<<VC2D(coverPoint,0.2,"magenta")<<flush);
     TGLOGPOL(1,<<_2D<<VC2D(coverPoint,0.3,"magenta")<<flush);
     TGLOGPOL(1,<<_2D<<VL2D(coverPoint,assumedNextOpponentPosition,"magenta")<<flush);
    
    targetPosition.clone( coverPoint );
  //end of computation
  return targetPosition;
}

//============================================================================
// getAttackOrientedPosition
//============================================================================
/**
 * getAttackOrientedPosition computes a good position for a defending
 * player (i.e. for a defender) in case our team is actually attacking.
 * (Under those circumstances this player should of course support the
 * attack to some extent.)
 * @ballApproachment determines to which degree should I take the current
 * ball position into consideration.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getAttackOrientedPosition(float ballApproachment)
{
  //start the computation of an appropriate position to drag the match forward
  OpponentAwarePosition targetPosition;
  
  //int myRole = cvFormation.get_role(WSinfo::me->number);//not needed
  
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    targetPosition.opponentNumber 
      = cvFormation.getDirectOpponentNumber(WSinfo::me->number);
    //1. basierend auf aktueller opponentPosition berechne die voraussichtliche
    //   position im naechsten zyklus ==> alpha
      const Vector currentOpponentPosition = directOpponent->pos; 
      Vector tempOpponentVelocity = directOpponent->vel;
      if ( directOpponent->age_vel > 3 )
      { //keine information ueber die geschwindigkeit des gegners -> 0/0
        tempOpponentVelocity.setXY( 0.0, 0.0 );
      }
      Vector assumedNextOpponentPosition 
        = currentOpponentPosition + tempOpponentVelocity;
      
     TGLOGPOL(1,<<_2D<<VC2D(assumedNextOpponentPosition,0.3,"4444ff")<<flush);

    //2. berechne punkt auf der strecke "alpha bis aktuelle ballposition", der
    //   10% von alpha entfernt ist ==> gamma
    Vector newBallPos =   WSinfo::ball->pos 
                        + ServerOptions::ball_decay * WSinfo::ball->vel;
    Vector coverPoint = (1.0-ballApproachment) * (assumedNextOpponentPosition
                                                  - newBallPos ) + newBallPos;
    TGLOGPOL(1,<<_2D<<VL2D(coverPoint,WSinfo::me->pos,"magenta")<<flush);
    TGLOGPOL(1,<<_2D<<VC2D(coverPoint,0.2,"ff9900")<<flush);

    targetPosition.clone( coverPoint );

  }
  else
  {
    //according to my formation, i currently have no direct opponent
    TGLOGPOL(0,<<"NO OPPONENT FOUND => go to home position"<<flush);
    targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  }
  return targetPosition;
}

//============================================================================
// getFormationAndDirectOpponentAwareStrategicPosition
//============================================================================
/**
 */
OpponentAwarePosition 
OpponentAwarePositioning::getFormationAndDirectOpponentAwareStrategicPosition()
{
  OpponentAwarePosition targetPosition;
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    targetPosition.opponentNumber 
      = cvFormation.getDirectOpponentNumber(WSinfo::me->number);
    //1. basierend auf aktueller opponentPosition berechne die voraussichtliche
    //   position im naechsten zyklus ==> alpha
      const Vector currentOpponentPosition = directOpponent->pos; 
      Vector tempOpponentVelocity = directOpponent->vel;
      if ( directOpponent->age_vel > 3 )
      { //keine information ueber die geschwindigkeit des gegners -> 0/0
        tempOpponentVelocity.setXY( 0.0, 0.0 );
      }
      Vector assumedNextOpponentPosition 
        = currentOpponentPosition + tempOpponentVelocity;
      TGLOGPOL(1,<<_2D<<VC2D(assumedNextOpponentPosition,0.3,"4444ff")<<flush);
    //2. berechne eine position, die zu meiner rolle in der aufstellung passt
    Vector formationBasedPosition;
    PPlayer supportedPlayer = NULL, mySupportingPlayer = NULL;
    switch (WSinfo::me->number)
    {
      case 2: case 3: case 4: case 5: case 9: case 10: case 11:
      {
        formationBasedPosition = getHomePosition();
        break;
      }
      case 6: case 7: case 8:
      {
        if ( getSupportedAndSupportingPlayerForMidfielder
               ( WSinfo::me->number,
                 supportedPlayer,
                 mySupportingPlayer ) )
        {
          TGLOGPOL(2,"OpponentAwarePosition: Dynamic supporting/supported teammates.");
        }
        else
        {
          if (WSinfo::me->number == 6)
          {
            supportedPlayer = WSinfo::get_teammate_by_number( 9 );
            mySupportingPlayer = WSinfo::get_teammate_by_number( 2 );
          }
          else
          if (WSinfo::me->number == 7)
          {
            supportedPlayer = WSinfo::get_teammate_by_number( 10 );
            mySupportingPlayer = WSinfo::get_teammate_by_number( 3 );
          }
          else
          if (WSinfo::me->number == 8)
          {
            supportedPlayer = WSinfo::get_teammate_by_number( 11 );
            mySupportingPlayer = WSinfo::get_teammate_by_number( 5 );
          }
        }
      }
    }
    if (supportedPlayer)
    {
      formationBasedPosition = supportedPlayer->pos;
      formationBasedPosition.subFromX( 10.0 );
      if (   mySupportingPlayer  )
        formationBasedPosition.setY(
            Tools::max(
              0.5*supportedPlayer->pos.getY() + 0.5*mySupportingPlayer->pos.getY(),
              supportedPlayer->pos.getY()) );
      if (   mySupportingPlayer
          && formationBasedPosition.getX() < mySupportingPlayer->pos.getX() + 5.0 )
        formationBasedPosition.setX( 0.6*supportedPlayer->pos.getX() + 0.4*mySupportingPlayer->pos.getX() );
      if (WSinfo::ball->pos.getX() > supportedPlayer->pos.getX() + 3.0)
        formationBasedPosition.addToX( 5.0 );
      if (WSinfo::me->number==6 && formationBasedPosition.getY()<10.0)
        formationBasedPosition.setY( 10.0 );
      if (WSinfo::me->number==8 && formationBasedPosition.getY()>-10.0)
        formationBasedPosition.setY( -10.0 );
    }
    else
      formationBasedPosition = getHomePosition();
    //3. ballberuecksichtigung
    Vector newBallPos =   WSinfo::ball->pos 
                        + ServerOptions::ball_decay * WSinfo::ball->vel;
    //4. gewichtung
    double weightForBall=1.0, weightForFormation=1.0, weightForDirectOpponent=1.0;
    if (getRole(WSinfo::me->number) == PT_DEFENDER)
    {
      TGLOGPOL(0,"WARNING: I am defender, it seems not very useful to go to"
        <<" formation-based strategic position. I should be more careful.");
      weightForDirectOpponent = 0.7;
      weightForFormation      = 0.0;
      weightForBall           = 0.3;
    }
    if (getRole(WSinfo::me->number) == PT_MIDFIELD)
    {
      weightForDirectOpponent = 0.5;
      weightForFormation      = 0.4;
      weightForBall           = 0.1;
    }
    if (getRole(WSinfo::me->number) == PT_ATTACKER)
    {
      weightForDirectOpponent = 0.4;
      weightForFormation      = 0.4;
      weightForBall           = 0.2;
    }
    targetPosition.setX(   weightForDirectOpponent * assumedNextOpponentPosition.getX()
                         + weightForFormation      * formationBasedPosition.getX()
                         + weightForBall           * newBallPos.getX() );
    targetPosition.setY(   weightForDirectOpponent * assumedNextOpponentPosition.getY()
                         + weightForFormation      * formationBasedPosition.getY()
                         + weightForBall           * newBallPos.getY() );
    TGLOGPOL(0,<<_2D<<VL2D(WSinfo::me->pos,
                      assumedNextOpponentPosition,"777777"));
    TGLOGPOL(0,<<_2D<<VL2D(WSinfo::me->pos,
                      formationBasedPosition,"ffff33"));
    TGLOGPOL(0,<<_2D<<VL2D(WSinfo::me->pos,
                      newBallPos,"ff8c00"));
  }
  else
  {
    //according to my formation, i currently have no direct opponent
    TGLOGPOL(0,<<"NO OPPONENT FOUND => go to home position"<<flush);
    targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  }
  if (targetPosition.getX() > WSinfo::his_team_pos_of_offside_line()-0.5)
    targetPosition.setX( WSinfo::his_team_pos_of_offside_line()-0.5 );
  return targetPosition;
}

//============================================================================
// getOpponentAwareDefensiveFallbackHomePosition
//============================================================================
bool
OpponentAwarePositioning::getSupportedAndSupportingPlayerForMidfielder
                          ( int       midfielderNumber,
                            PPlayer & supportedPlayer,
                            PPlayer & supportingPlayer )
{
  if (midfielderNumber != 6 && midfielderNumber != 7 && midfielderNumber != 8)
    return false;
  //defaults
  switch (midfielderNumber)
  {
    case 6: 
    {
      supportedPlayer = WSinfo::get_teammate_by_number( 9 );
      supportingPlayer = WSinfo::get_teammate_by_number( 2 );
      break;
    }
    case 7: 
    {
      supportedPlayer = WSinfo::get_teammate_by_number( 10 );
      supportingPlayer = WSinfo::get_teammate_by_number( 3 );
      break;
    }
    case 8:
    {
      supportedPlayer = WSinfo::get_teammate_by_number( 11 );
      supportingPlayer = WSinfo::get_teammate_by_number( 5 );
      break;
    }
  }
  //current
  PlayerSet midfieldersDirectOpponents;
  PPlayer aDirectOpponent = getDirectOpponent( 6 );
  if ( aDirectOpponent ) midfieldersDirectOpponents.append(aDirectOpponent);
  aDirectOpponent = getDirectOpponent( 7 );
  if ( aDirectOpponent ) midfieldersDirectOpponents.append(aDirectOpponent);
  aDirectOpponent = getDirectOpponent( 8 );
  if ( aDirectOpponent ) midfieldersDirectOpponents.append(aDirectOpponent);
  midfieldersDirectOpponents.keep_and_sort_players_by_y_from_right(
    midfieldersDirectOpponents.num);
  if (midfieldersDirectOpponents.num != 3)
    return false;
  PPlayer 
    responsibleTeammateLeft 
      = getResponsiblePlayerForOpponent(midfieldersDirectOpponents[0]->number), 
    responsibleTeammateCenter
      = getResponsiblePlayerForOpponent(midfieldersDirectOpponents[1]->number),
    responsibleTeammateRight
      = getResponsiblePlayerForOpponent(midfieldersDirectOpponents[2]->number);
  if (   responsibleTeammateLeft
      && responsibleTeammateCenter
      && responsibleTeammateRight )
  {
    if (midfielderNumber == responsibleTeammateLeft->number)
    {
      supportedPlayer  = WSinfo::get_teammate_by_number( 9 );
      supportingPlayer = WSinfo::get_teammate_by_number( 2 );
      TGLOGPOL(3,"Support for "<<midfielderNumber<<": 9 / 2");
      return true;
    }
    if (midfielderNumber == responsibleTeammateCenter->number)
    {
      supportedPlayer  = WSinfo::get_teammate_by_number(10 );
      supportingPlayer = WSinfo::get_teammate_by_number( 3 );
      TGLOGPOL(3,"Support for "<<midfielderNumber<<": 10 / 3");
      return true;
    }
    if (midfielderNumber == responsibleTeammateRight->number)
    {
      supportedPlayer  = WSinfo::get_teammate_by_number(11 );
      supportingPlayer = WSinfo::get_teammate_by_number( 5 );
      TGLOGPOL(3,"Support for "<<midfielderNumber<<": 11 / 5");
      return true;
    }
  }
  return false;
}

//============================================================================
// getOpponentAwareDefensiveFallbackHomePosition
//============================================================================
/**
 * It may happen that a player loses his direct opponent. Then, it is 
 * advisable to move to some kind of fallback position. Note, that the 
 * home position is not a good idea then. Instead my direct opponent may
 * already have approached my goal. That is why, this fallback position
 * lies nearer towards my goal, i.e. the fallback position is a rather
 * defensive position.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getOpponentAwareDefensiveFallbackHomePosition()
{
  OpponentAwarePosition targetPosition;
  int myRole = cvFormation.get_role(WSinfo::me->number);
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(WSinfo::me->number, directOpponent) )
  {
    targetPosition.opponentNumber 
      = cvFormation.getDirectOpponentNumber(WSinfo::me->number);

    double yOffset = 0.0, xOffset = 0.0;
    if (WSinfo::me->pos.getY() > 7.0) yOffset = 7.0;
    if (WSinfo::me->pos.getY() <-7.0) yOffset = -7.0;
    if (fabs(WSinfo::me->pos.getY()) < 7.0) yOffset = WSinfo::me->pos.getY();
    double standardX = WSinfo::my_team_pos_of_offside_line(),
          standardXHis = WSinfo::his_team_pos_of_offside_line();
  
    switch (myRole)
    {
      //case_PT_SWEEPER:
      case PT_DEFENDER:
        yOffset *= 1.0;
        xOffset = standardX - 1.0;
        if (WSinfo::me->pos.getX() - xOffset < 2.0) xOffset = standardX;
      break;
      case PT_MIDFIELD:
        yOffset *= 1.5;
        xOffset = 0.6*standardX + 0.4*standardXHis;
      break;
      case PT_ATTACKER:
        yOffset *= 2.0;
        xOffset = 0.3*standardX + 0.7*standardXHis;
      break;
      default:
        TGLOGPOL(0,<<"WARNING: OpponentAwarePositioning: Unsopported player role."<<flush);
    }
    targetPosition.setXY( xOffset, yOffset );
  }
  else
  {
    //according to my formation, i currently have no direct opponent
    TGLOGPOL(0,<<"WARNING: NO OPPONENT FOUND => go to home position"<<flush);
    targetPosition.clone( cvFormation.getHomePosition(WSinfo::me->number) );
  }
  return targetPosition;
}

//============================================================================
// getOpponentAwareMyKickInPosition
//============================================================================
/**
 * getOpponentAwareMyKickInPosition computes an appropriate position to 
 * which I should move, if we have a kick in.
 */
OpponentAwarePosition 
OpponentAwarePositioning::getOpponentAwareMyKickInPosition()
{
  if (WSinfo::ball->pos.distance(WSinfo::me->pos) < 20.0)
    return getAttackOrientedPosition(0.4);
  else
    return getAttackOrientedPosition();
}
      
//============================================================================
// getBreakthroughPlayerStopPosition
//============================================================================
/**
 * This method is crucial, in particular for the sweeper. It may happen that
 * an opponent player breaks through, i.e. moves beyond our offside line and
 * is approaching our goal. Then, high danger is given. For this special 
 * situation we need to find a position which enables our player to best
 * stop/intercept the "through-breaking" opponent btp.
 */
OpponentAwarePosition
OpponentAwarePositioning::getBreakthroughPlayerStopPosition(PPlayer btp,
                                                            bool  areWeAttacking)
{
  OpponentAwarePosition returnValue;
  Vector targetPosition, delta, goalReferencePoint;
  goalReferencePoint.setX( -FIELD_BORDER_X + 1.0 );
  goalReferencePoint.setY(  btp->pos.getY() + btp->vel.getY() );
  if (goalReferencePoint.getY() > 7.0) goalReferencePoint = MY_GOAL_LEFT_CORNER;
  if (goalReferencePoint.getY() <-7.0) goalReferencePoint = MY_GOAL_RIGHT_CORNER;
  double realDecay = 0.9;
  delta = (btp->pos+btp->vel) - goalReferencePoint;
  if (areWeAttacking == false)
  {
    double decayMin = 0.5, decayMax = 0.7, decayDelta = decayMax-decayMin;
    if (     0
          && WSinfo::me->number != 4 //MY_SWEEPER_NUMBER
          && WSinfo::ball->pos.distance(MY_GOAL_CENTER) < 20.0
          && btp->age_ang < 2 && btp->age_vel < 2
          && fabs( btp->ang.get_value_mPI_pPI() ) < PI/1.5
          && btp->vel.getX() > -0.2 )
    { decayMin = 0.75; decayMax = 0.85; decayDelta = decayMax-decayMin; }
    realDecay = Tools::min( delta.norm() / (FIELD_BORDER_X*0.75), 1.0);
    realDecay = decayDelta*realDecay + decayMin;
    delta *= realDecay;
  }
  else //we are attacking!
  {
    delta *= realDecay;
    double multiplier = 1.0;
    Vector dummyTarget = goalReferencePoint + delta;
    PlayerSet teammates = WSinfo::valid_teammates_without_me;
    PPlayer myGoalie = WSinfo::get_teammate_by_number(WSinfo::ws->my_goalie_number);
    teammates.remove( myGoalie );
    teammates.keep_and_sort_players_by_x_from_left(1);
    double myOffSideLineWhenNotConsideringMe;
    if (teammates.num > 0)
      myOffSideLineWhenNotConsideringMe = teammates[0]->pos.getX();
    else
      myOffSideLineWhenNotConsideringMe = WSinfo::my_team_pos_of_offside_line();
      
    if ((dummyTarget).getX() > myOffSideLineWhenNotConsideringMe)
    {
      multiplier = (myOffSideLineWhenNotConsideringMe - goalReferencePoint.getX())
                   / ( dummyTarget.getX() - goalReferencePoint.getX());
      delta *= multiplier;
    }
  }
  targetPosition = goalReferencePoint + delta;
  returnValue.clone( targetPosition );
  TGLOGPOL(0,<<"getBreakthroughPlayerStopPosition = "<<returnValue.getX()<<", "<<returnValue.getY()<<"  realDecay="<<realDecay<<flush);
  return returnValue;
}

//============================================================================
// getRunningDuelTargetPosition
//============================================================================
/**
 */
OpponentAwarePosition
OpponentAwarePositioning::getRunningDuelTargetPosition(PPlayer rpl)
{
  OpponentAwarePosition returnValue;
  double xPositionFromWhereTheOpponentMayScore = -FIELD_BORDER_X + 11.0;
  if (rpl->pos.getX() < -FIELD_BORDER_X + 15.0)
    xPositionFromWhereTheOpponentMayScore = -FIELD_BORDER_X + 8.0;
  Vector targetPosition
    = Tools::point_on_line( Vector( rpl->ang ), //steigung  
                            rpl->pos,
                            xPositionFromWhereTheOpponentMayScore );
  if (targetPosition.getY() > 9.0) targetPosition.setY(  9.0 );
  if (targetPosition.getY() <-9.0) targetPosition.setY( -9.0 );

  returnValue.clone( targetPosition );
  TGLOGPOL(0,<<"getRunningDuelTargetPosition = "<<returnValue.getX()<<", "
    <<returnValue.getY()<<flush);
  return returnValue;
}

//============================================================================
// getStrategicSweeperPosition
//============================================================================
/**
 */
OpponentAwarePosition
OpponentAwarePositioning::getStrategicSweeperPosition(PPlayer btp,
                                                      bool  areWeAttacking)
{ 
  OpponentAwarePosition returnValue;
  Vector targetPosition, delta, goalReferencePoint;
  goalReferencePoint.setX( -FIELD_BORDER_X );
  goalReferencePoint.setY( btp->pos.getY() + btp->vel.getY() );
  if (goalReferencePoint.getY() > 7.0) goalReferencePoint = MY_GOAL_LEFT_CORNER;
  if (goalReferencePoint.getY() <-7.0) goalReferencePoint = MY_GOAL_RIGHT_CORNER;
  delta = (btp->pos+btp->vel) - goalReferencePoint;
  if (areWeAttacking == false)
  {
    double decayMin = 0.5, decayMax = 0.8, decayDelta = decayMax-decayMin;
    double realDecay = Tools::min( delta.norm() / (FIELD_BORDER_X*0.5), 1.0);
    realDecay = decayDelta*realDecay + decayMin;
    delta *= realDecay;
  }
  else //we are attacking!
  {
    delta *= 0.9;
    double multiplier = 1.0;
    Vector dummyTarget = goalReferencePoint + delta;
    PlayerSet teammates = WSinfo::valid_teammates_without_me;
    PPlayer myGoalie = WSinfo::get_teammate_by_number(WSinfo::ws->my_goalie_number);
    teammates.remove( myGoalie );
    teammates.keep_and_sort_players_by_x_from_left(1);
    double myOffSideLineWhenNotConsideringMe;
    if (teammates.num > 0)
      myOffSideLineWhenNotConsideringMe = teammates[0]->pos.getX();
    else
      myOffSideLineWhenNotConsideringMe = WSinfo::my_team_pos_of_offside_line();
      
    if ((dummyTarget).getX() > myOffSideLineWhenNotConsideringMe)
    {
      multiplier = (myOffSideLineWhenNotConsideringMe - goalReferencePoint.getX())
                   / ( dummyTarget.getX() - goalReferencePoint.getX());
      delta *= multiplier;
    }
  }
  targetPosition = goalReferencePoint + delta;
  returnValue.clone( targetPosition );
  TGLOGPOL(0,<<"getStrategicSweeperPosition = "<<returnValue.getX()<<", "<<returnValue.getY()<<flush);
  return returnValue;
}

//============================================================================
// computeDirectOpponents
//============================================================================
/**
 * This method gathers and stores all information that is relevant regarding
 * the player's direct opponent.
 */
void
OpponentAwarePositioning::updateDirectOpponentInformation()
{
  cvFormation.computeDirectOpponents();
}

//============================================================================
// getResponsiblePlayerForOpponent
//============================================================================
/**
 * This method calculates which teammate is responsible for the opponent
 * player with number.
 */
PPlayer
OpponentAwarePositioning::getResponsiblePlayerForOpponent(int number)
{
  PPlayer returnValue = NULL;
  cvFormation.getResponsiblePlayerForOpponent(number, returnValue);
  return returnValue;
}

//============================================================================
// getDirectOpponent
//============================================================================
/**
 * This method returns my direct opponent.
 */
PPlayer
OpponentAwarePositioning::getDirectOpponent(int number)
{
  PPlayer returnValue = NULL;
  cvFormation.getDirectOpponent(number, returnValue);
  return returnValue;
}

//============================================================================
// setOpponentAwareRelevantTeammates
//============================================================================
/**
 * This method can be used to influence the attention (i.e. the listening
 * behavior) of me. Depending on my number this method decides to which 
 * of my teammates I should pay attention (listening).
 * This method may be called if I have lost my direct opponent, for example.
 * Then, my teammates can communicate to me the position of my direct
 * opponent.
 * 
 * NOTE: This feature has not yet been tested sufficiently (TG, 05/05).
 */
void     
OpponentAwarePositioning::setOpponentAwareRelevantTeammates()
{
  double myDistToMyGoal = WSinfo::me->pos.distance(MY_GOAL_CENTER);
  double hearThreshold  = 50.0; //ServerOtpions::audio_cut_dist
  switch(WSinfo::me->number)
  {
    case 2:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 4, 6, 3, 1, 3, 9);
      else
        WSinfo::set_relevant_teammates( 4, 6, 3, 3, 9);
      break;
    case 3:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 4, 7, 2, 1, 5, 10);
      else
        WSinfo::set_relevant_teammates( 4, 7, 2, 1, 5, 10);
      break;
    case 4:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 1, 3, 2, 5);
      else
        WSinfo::set_relevant_teammates( 3, 2, 5);
      break;
    case 5:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 4, 8, 3, 1, 11);
      else
        WSinfo::set_relevant_teammates( 4, 8, 3, 11);
      break;
    case 6:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 4, 9, 7, 1, 2);
      else
        WSinfo::set_relevant_teammates( 4, 9, 7, 2);
      break;
    case 7:
      WSinfo::set_relevant_teammates( 4, 10, 6, 8, 3);
      break;
    case 8:
      if (myDistToMyGoal < hearThreshold)
        WSinfo::set_relevant_teammates( 4, 11, 7, 1, 5);
      else
        WSinfo::set_relevant_teammates( 4, 11, 7, 5);
      break;
    case 9:
      WSinfo::set_relevant_teammates( 4, 2, 7, 6, 3);
      break;
    case 10:
      WSinfo::set_relevant_teammates( 4, 3, 7, 6, 8);
      break;
    case 11:
      WSinfo::set_relevant_teammates( 4, 5, 7, 8, 3);
      break;
  }  
}

//============================================================================
// getDistanceFromPlayerToDirectOpponent
//============================================================================
/**
 * This method calculates the distance between a specified teammate player
 * and his direct opponent player.
 */
bool 
OpponentAwarePositioning::getDistanceFromPlayerToDirectOpponent
                          ( PPlayer teammate,
                            double & distance )
{
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent( teammate->number, directOpponent) )
  {
    distance = (teammate->pos - directOpponent->pos).norm();
    return true;
  }
  distance = 0.0;
  return false;
}

//============================================================================
// getDirectOpponentHeadStartToMyGoal
//============================================================================
/**
 * This method calculates the 'head start' of a teammate's direct opponent
 * player. This measure tells how much nearer the direct opponent is towards
 * my goal than my teammate (who is responsible for that opponent).
 */
bool 
OpponentAwarePositioning::getDirectOpponentHeadStartToMyGoal
                          ( PPlayer teammate,
                            double & distance )
{
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent( teammate->number, directOpponent) )
  {
    double myDist2MyGoal = (teammate->pos-MY_GOAL_CENTER).norm();
    double dirOppDist2MyGoal = (directOpponent->pos-MY_GOAL_CENTER).norm();
    distance = myDist2MyGoal - dirOppDist2MyGoal;
    return true;
  }
  distance = 0.0;
  return false;
}

//============================================================================
// getViableSurveillanceOpponentsForPlayer
//============================================================================
/**
 * This method represents the 'opposite' of method 
 * setOpponentAwareRelevantTeammates. By this method the numbers of opponent
 * players are defined which can/may be monitored and whose positions may
 * be communidated (bradcasted). This is useful for telling some teammate
 * the current position of his direct opponent.
 * 
 * NOTE: This feature has not yet been tested sufficiently.
 */
PlayerSet
OpponentAwarePositioning::getViableSurveillanceOpponentsForPlayer(PPlayer teammate)
{
  PlayerSet returnValue;
  int numberOfPlayersSurveilled = 0;
  int numbersOfSurveilledPlayer[NUM_PLAYERS-1];
  switch (teammate->number)
  {
    case 1:
      numbersOfSurveilledPlayer[ 0 ] = 3;
      numbersOfSurveilledPlayer[ 1 ] = 2;
      numbersOfSurveilledPlayer[ 2 ] = 5;
      numbersOfSurveilledPlayer[ 3 ] = 7;
      numbersOfSurveilledPlayer[ 4 ] = 6;
      numbersOfSurveilledPlayer[ 5 ] = 8;
      numberOfPlayersSurveilled = 6;
    break;
    case 2:
      numbersOfSurveilledPlayer[ 0 ] = 3;
      numbersOfSurveilledPlayer[ 1 ] = 6;
      numbersOfSurveilledPlayer[ 2 ] = 9;
      numberOfPlayersSurveilled = 3;
    break;
    case 3:
      numbersOfSurveilledPlayer[ 0 ] = 2;
      numbersOfSurveilledPlayer[ 1 ] = 5;
      numbersOfSurveilledPlayer[ 2 ] = 7;
      numbersOfSurveilledPlayer[ 3 ] = 10;
      numbersOfSurveilledPlayer[ 4 ] = 9;
      numbersOfSurveilledPlayer[ 5 ] = 11;
      numberOfPlayersSurveilled = 6;
    break;
    case 4:
      numbersOfSurveilledPlayer[ 0 ] = 3;
      numbersOfSurveilledPlayer[ 1 ] = 2;
      numbersOfSurveilledPlayer[ 2 ] = 5;
      numbersOfSurveilledPlayer[ 3 ] = 7;
      numbersOfSurveilledPlayer[ 4 ] = 6;
      numbersOfSurveilledPlayer[ 5 ] = 8;
      numbersOfSurveilledPlayer[ 6 ] = 10;
      numbersOfSurveilledPlayer[ 7 ] = 9;
      numbersOfSurveilledPlayer[ 8 ] = 11;
      numberOfPlayersSurveilled = 9;
    break;
    case 5:
      numbersOfSurveilledPlayer[ 0 ] = 3;
      numbersOfSurveilledPlayer[ 1 ] = 8;
      numbersOfSurveilledPlayer[ 2 ] = 11;
      numberOfPlayersSurveilled = 3;
    break;
    case 6:
      numbersOfSurveilledPlayer[ 0 ] = 2;
      numbersOfSurveilledPlayer[ 1 ] = 3;
      numbersOfSurveilledPlayer[ 2 ] = 7;
      numbersOfSurveilledPlayer[ 3 ] = 9;
      numberOfPlayersSurveilled = 4;
    break;
    case 7:
      numbersOfSurveilledPlayer[ 0 ] = 3;
      numbersOfSurveilledPlayer[ 1 ] = 2;
      numbersOfSurveilledPlayer[ 2 ] = 5;
      numbersOfSurveilledPlayer[ 3 ] = 6;
      numbersOfSurveilledPlayer[ 4 ] = 8;
      numbersOfSurveilledPlayer[ 5 ] = 10;
      numbersOfSurveilledPlayer[ 6 ] = 9;
      numbersOfSurveilledPlayer[ 7 ] = 11;
      numberOfPlayersSurveilled = 8;
    break;
    case 8:
      numbersOfSurveilledPlayer[ 0 ] = 5;
      numbersOfSurveilledPlayer[ 1 ] = 3;
      numbersOfSurveilledPlayer[ 2 ] = 7;
      numbersOfSurveilledPlayer[ 3 ] = 11;
      numberOfPlayersSurveilled = 4;
    break;
    case 9:
      numbersOfSurveilledPlayer[ 0 ] = 6;
      numbersOfSurveilledPlayer[ 1 ] = 2;
      numberOfPlayersSurveilled = 2;
    break;
    case 10:
      numbersOfSurveilledPlayer[ 0 ] = 7;
      numbersOfSurveilledPlayer[ 1 ] = 3;
      numberOfPlayersSurveilled = 2;
    break;
    case 11:
      numbersOfSurveilledPlayer[ 0 ] = 8;
      numbersOfSurveilledPlayer[ 1 ] = 5;
      numberOfPlayersSurveilled = 2;
    break;
  }
  for (int i=0; i<numberOfPlayersSurveilled; i++)
  {
    const int surveilledTeammatesAndOpponentsMaxAge = 1;
    PPlayer surveilledTeammate;
    if ( WSinfo::get_teammate( numbersOfSurveilledPlayer[i], surveilledTeammate ) )
    {
      double distanceToDirectOpponent,
             directOpponentHeadStart;
      bool b1 = getDistanceFromPlayerToDirectOpponent ( surveilledTeammate,
                                                        distanceToDirectOpponent ),
           b2 = getDirectOpponentHeadStartToMyGoal ( surveilledTeammate,
                                                     directOpponentHeadStart );
      if ( b1 && b2 )
      {
if (WSinfo::me->number==1)
{
  //goalie output
  TGLOGPOL(1,<<"surveilledTeamm="<<numbersOfSurveilledPlayer[i]<<"  distanceToDirectOpponent="<<distanceToDirectOpponent<<"  directOpponentHeadStart="<<directOpponentHeadStart);
}
        if ( (    ( 
                    ( distanceToDirectOpponent > 5.0 && directOpponentHeadStart > 3.0 )
                    ||
                    (    WSinfo::ball->pos.distance(MY_GOAL_CENTER) < 25.0
                      && (distanceToDirectOpponent > 4.0 || directOpponentHeadStart > 2.5)
                    )
                  )
               && surveilledTeammate->age <= surveilledTeammatesAndOpponentsMaxAge 
               && surveilledTeammate->pos.distance(WSinfo::me->pos) < 30.0
             )
             ||
             (    ( 
                    ( distanceToDirectOpponent > 4.0 || directOpponentHeadStart > 2.5 )
                    ||
                    (    WSinfo::ball->pos.distance(MY_GOAL_CENTER) < 25.0
                      && (distanceToDirectOpponent > 3.0 || directOpponentHeadStart > 1.5)
                    )
                  )
               && surveilledTeammate->age == 0
               && surveilledTeammate->pos.distance(WSinfo::me->pos) < 30.0
             )
           )
        {
          PPlayer directOpponent;
          if (cvFormation.getDirectOpponent( numbersOfSurveilledPlayer[i], directOpponent)
              && directOpponent->age <= surveilledTeammatesAndOpponentsMaxAge )
            returnValue.append( directOpponent );
        }
      }
    }
  }
  return returnValue;
}

//in test state
/*
OpponentAwarePosition
OpponentAwarePositioning::getAttackOrientedPassReceivingPosition()
{
}
*/

//############################################################################
// In the following, methods for attack positioning shall be specified.
//############################################################################
double OpponentAwarePositioning::cvCriteriaWeights[4][MAX_CRITERIA]
  = {
      //xPos, bDis, ddoD, stam, passw
//      { 0.21, 0.06, 0.29, 0.21, 0.1},  //PT_DEFENDER=0
//      { 0.2,  0.3,  0.2,  0.0,  0.4},   //PT_MIDFIELD=1
//      { 0.3,  0.2,  0.1,  0.0,  0.3},   //PT_ATTACKER=2
//      { 0.0,  0.0,  0.0,  0.0,  0.0}    //PT_GOALIE=3
      { 0.21, 0.06, 0.29, 0.21, 0.1},  //PT_DEFENDER=0
      { 0.4,  0.2,  0.1,  0.05, 0.3},   //PT_MIDFIELD=1
      { 0.3,  0.2,  0.1,  0.05, 0.3},   //PT_ATTACKER=2
      { 0.0,  0.0,  0.0,  0.0,  0.0}    //PT_GOALIE=3
    };

void   
OpponentAwarePositioning::setInterceptInformation
                          (tInterceptInformation & icptInfo)
{
  cvInterceptInformation = icptInfo;
}

double
OpponentAwarePositioning::getBestOfferingPosition( Vector & offeringPosition )
{
  return getBestOfferingPosition( WSinfo::me, offeringPosition );
}

double
OpponentAwarePositioning::getBestOfferingPosition(PPlayer  teammate,
                                                  Vector & offeringPosition)
{
  if (!teammate) return -1.0;
  int numberOfCriteria = MAX_CRITERIA;
#if LOGGING && BASIC_LOGGING
  double bestCriteriaValues[numberOfCriteria];
#endif
  double criteriaValues [numberOfCriteria];
  double valueForTargetPosition = 0.0;
  double bestValueForTargetPosition = -1.0;
  double targetDistance;
  Angle  targetAngle;
  Vector targetPosition,
         bestTargetPosition;
  //value init
  for (int i=0; i<numberOfCriteria; i++) criteriaValues[i] = 0.0;
#if LOGGING && BASIC_LOGGING
  for (int i=0; i<numberOfCriteria; i++) bestCriteriaValues[i] = 0.0;
#endif
  //start calculations
  PPlayer passGivingTeammate = NULL;
  Vector passStartPosition 
    = calculateExpectedPassStartPosition(passGivingTeammate);
  double minDist  =  2.0,
        maxDist  =  12.0,                
        distStep =  1.0;
  if (passGivingTeammate)
  {
    WSinfo::me->pos.distance(passGivingTeammate->pos);
    maxDist += passGivingTeammate->pos.distance(passStartPosition);
  }
  if (maxDist > 15.0) maxDist = 15.0;
  if (maxDist <  5.0) maxDist =  5.0;
  TGLOGPOL(0, <<_2D<<VC2D(WSinfo::me->pos, maxDist, "22ff22"));
  Angle minAng   =  0.0,
        maxAng   = 2*PI,
        angStep  = DEG2RAD(15);
Vector dummyPredIcpt, bestDummyPredIcpt;

  //HAUPTSCHLEIFE: ueber alle positionen, an denen ich mich anbieten kann
  for ( int i=0; i <= (maxDist-minDist)/distStep; i++)
  {
    targetDistance = minDist + i*distStep;
    for (int j=0; j < (maxAng-minAng)/angStep; j++)
    {
      targetAngle = minAng + j*angStep;
      targetPosition.setX( targetDistance * cos(targetAngle) );
      targetPosition.setY( targetDistance * sin(targetAngle) );
      targetPosition += teammate->pos;

      if (OpponentAwarePositioning::shallIExcludeOfferPositionFromConsideration
                                    (teammate, targetPosition, 
                                     passStartPosition, passGivingTeammate))
      {
        if (0&&(float)rand()/(float)RAND_MAX < 0.01) {
          TGLOGPOL(0, _2D << VSTRING2D( targetPosition,
                                       "x",
                                       "ffff66" ));
        }
        continue;
      }
      //main function call: we evaluate a single position
      valueForTargetPosition = evaluateOfferingPosition( teammate,
                                                         targetPosition,
                                                         criteriaValues);
      if (valueForTargetPosition < 0.0) continue; //position excluded
      
      //TGLOGPOL(0, "OFFPOSEVAL: " << targetPosition-teammate->pos << ": "<<((int)(valueForTargetPosition*1000))/10.0);
if (0&&(float)rand()/(float)RAND_MAX < 0.01) {
  TGLOGPOL(0, _2D << VSTRING2D( targetPosition,
                               ((int)(valueForTargetPosition*1000))/10.0,
                               "ffff66" ));
}
      if (valueForTargetPosition > bestValueForTargetPosition)
      {
        bestValueForTargetPosition = valueForTargetPosition;
        bestTargetPosition = targetPosition;
#if LOGGING && BASIC_LOGGING
        for (int k=0; k<numberOfCriteria; k++)
          bestCriteriaValues[k] = criteriaValues[k];
#endif
      }
    }
  }

  if (bestValueForTargetPosition >= 0.0)
  {
  offeringPosition = bestTargetPosition;
  TGLOGPOL(0, _2D << VC2D(offeringPosition, 1.0, "ffcc00"));
  TGLOGPOL(0, _2D << VSTRING2D( offeringPosition,
                               ((int)(bestValueForTargetPosition*1000))/10.0,
                               "ffcc00" ));
for (int k = 0; k < numberOfCriteria; ++k) 
{
	TGLOGPOL(0, "BEST OFFPOS: criteria["<<k<<"] = "<<bestCriteriaValues[k] << "  w="
                       <<cvCriteriaWeights[ getRole(teammate->number) ][k]
                       <<"  -> "<<bestCriteriaValues[k]*cvCriteriaWeights[ getRole(teammate->number) ][k]);
}
Vector icptPoint;
double dummyFreeness;
OpponentAwarePositioning::isPassWayFreeEnoughForThisPosition
  ( teammate,
    bestTargetPosition,
    icptPoint,
    dummyFreeness,
    true);
TGLOGPOL(0, "    ==> "<< bestValueForTargetPosition<<" at "<<offeringPosition<<" (pass from "<<icptPoint<<")");
TGLOGPOL(0, <<_2D<<VL2D(icptPoint,
                       offeringPosition, "ffff66"));
  }
  if (bestValueForTargetPosition <= 0.0)
  {
    bestValueForTargetPosition = 0.0;
    TGLOGPOL(0, << "OpponentAwarePositioning: WARNING: Did not find any offering situation!");
  }
  return bestValueForTargetPosition;
}

//============================================================================
// getPenaltyAreaOfferingHomePosition()
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning::getPenaltyAreaOfferingHomePosition(int number)
{
  Vector returnValue
    = cvFormation.getHomePosition(number);
      //hard-coded
      switch (number)
      {
        case  6: returnValue.setX( FIELD_BORDER_X-13.0 ); returnValue.setY(  12.0 ); break;
        case  7: returnValue.setX( FIELD_BORDER_X-13.0 ); returnValue.setY(   0.0 ); break;
        case  8: returnValue.setX( FIELD_BORDER_X-13.0 ); returnValue.setY( -12.0 ); break;
        case  9: returnValue.setX( FIELD_BORDER_X- 7.0 ); returnValue.setY(   8.0 ); break;
        case 10: returnValue.setX( FIELD_BORDER_X- 7.0 ); returnValue.setY(   0.0 ); break;
        case 11: returnValue.setX( FIELD_BORDER_X- 7.0 ); returnValue.setY(  -8.0 ); break;
        default: break;
      }
  return returnValue;
}      

//============================================================================
// evaluatePenaltyAreaOfferingPosition
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::evaluatePenaltyAreaOfferingPosition 
  ( PPlayer  teammate,
    Vector & targetPosition,
    bool debug )
{
      //this calculation is made a little earlier (to find passStartPos)
      Vector passStartPos;
      double valueForTargetPosition_1;
      isPassWayFreeEnoughForThisPosition( teammate,
                                          targetPosition,
                                          passStartPos,
                                          valueForTargetPosition_1, //freenessScore,
                                          debug, //debug
                                          true); //considerationForPenaltyArea


      //######## EXCLUSION 1 ###########
      PlayerSet nearTeammates = WSinfo::valid_teammates_without_me;
      Vector extendedTargetPosition = targetPosition - teammate->pos;
      extendedTargetPosition.normalize( 1.5*(targetPosition - teammate->pos).norm()  );
      extendedTargetPosition += passStartPos;
      nearTeammates.keep_players_in_quadrangle(teammate->pos,
                                               extendedTargetPosition,
                                               3.0, 
                                               6.0);
      if (
             targetPosition.getX() > FIELD_BORDER_X - 1.0
          || targetPosition.getX() > WSinfo::his_team_pos_of_offside_line()
          || fabs(targetPosition.getY()) > FIELD_BORDER_Y - 0.3
          || nearTeammates.num > 0
          || (   targetPosition.getX() > FIELD_BORDER_X - 3.0
              && targetPosition.distance(WSinfo::ball->pos) > 15.0 )
         )
      {
        //TGLOGPOL(5, "EXCLUDE pos "<<targetPosition<<": too near to offside line / field border [mePos="<<teammate->pos<<",nearTmm.num="<<nearTeammates.num<<",hisOffsLine="<<WSinfo::his_team_pos_of_offside_line()<<"]");
        return -1.0;
      }
      //######## EXCLUSION 2 ###########
      for (int k=0; k<WSinfo::valid_teammates.num; k++)
      {
        PPlayer p = WSinfo::valid_teammates[k];
        if (   p != teammate
            && p->pos.distance( WSinfo::ball->pos ) < 20.0
            &&    p->pos.distance(targetPosition) 
                < teammate->pos.distance(targetPosition) )
        {
          //TGLOGPOL(5, "EXCLUDE pos "<<targetPosition<<": teammate can reach target faster");
          return -1.0;
        }
      }
      //######## EXCLUSION 3 ###########
      nearTeammates = WSinfo::valid_teammates_without_me;
      extendedTargetPosition = targetPosition - passStartPos;
      extendedTargetPosition.normalize( 1.25*(targetPosition - passStartPos).norm()  );
      extendedTargetPosition += passStartPos;
      nearTeammates.keep_players_in_quadrangle(
                        passStartPos,
                        extendedTargetPosition,
                        3.0, 
                        (extendedTargetPosition - passStartPos).norm() );
      if (   nearTeammates.num > 1
          || (targetPosition - passStartPos).norm() < 6.0 )/*TG06: 5.0*/
      {
        TGLOGPOL(5, "EXCLUDE pos "<<targetPosition<<": found a teammate in the potential passway or passway too short");
        return -1.0;
      }
      //######## EXCLUSION 4 ########### /*TG06:neu*/
      PlayerSet nearOpponents = WSinfo::valid_opponents;
      nearOpponents.keep_players_in_circle(targetPosition, 3.5);
      if ( nearOpponents.num > 0 )
      {
        TGLOGPOL(5, "EXCLUDE pos "<<targetPosition<<": found an opponent near target position");
        return -1.0;
      }
      //######## EXCLUSION 5 ########### /*TG17*/
      if (   teammate->pos.getX() < 15.0
          || targetPosition.getX() < 25.0)
      {
        TGLOGPOL(5, "EXCLUDE pos "<<targetPosition<<": it is too far behind to think of penalty area offering");
        return -1.0;
      }
      
      //continue main function call: we evaluate a single position
      double valueForTargetPosition_2
        = evaluateOfferingPositionRegardingBallDistance 
                                 (teammate, 
                                  targetPosition, 
                                  passStartPos, //NOTE: NOT BALL, BUT PASS START POS!
                                  true );  //weAreInPenaltyArea==true
      
      double valueForTargetPosition_3
        = evaluateOfferingPositionRegardingDistanceToOpponents
                                 ( teammate,targetPosition );
                                 
      double valueForTargetPosition_4
        = Tools::max(0.0,1.0-(targetPosition.distance(HIS_GOAL_CENTER) / 40.0));

      double valueForTargetPosition_5
        = evaluateOfferingPositionRegardingDistanceFromPenaltyAreaHomePos
                                 ( teammate,targetPosition );

      double valueForTargetPosition; //TG17?!
      if (OpponentAwarePositioning::cvUseVersion2017)
        valueForTargetPosition
        =   0.21 * valueForTargetPosition_1  //passway freeness
          + 0.16 * valueForTargetPosition_2  //dist to (ball) pass start pos
          + 0.11 * valueForTargetPosition_3  //dist to opponents
          + 0.36 * valueForTargetPosition_4  //dist from his goal
          + 0.18 * valueForTargetPosition_5; //dist from home pos
      else
        valueForTargetPosition
        =   0.25 * valueForTargetPosition_1  //passway freeness
          + 0.15 * valueForTargetPosition_2  //dist to (ball) pass start pos
          + 0.15 * valueForTargetPosition_3  //dist to opponents
          + 0.23 * valueForTargetPosition_4  //dist from his goal
          + 0.22 * valueForTargetPosition_5; //dist from home pos

if (debug) { TGLOGPOL(0," v1="<<valueForTargetPosition_1<<" v2="<<valueForTargetPosition_2<<" v3="<<valueForTargetPosition_3<<" v4="<<valueForTargetPosition_4
<<" v5="<<valueForTargetPosition_5<<" => v="<<valueForTargetPosition); }
  return valueForTargetPosition;
}

//============================================================================
// getBestPenaltyAreaOfferingPosition
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::getBestPenaltyAreaOfferingPosition(PPlayer  teammate,
                                                             Vector & offeringPosition)
{
  if (!teammate) return -1.0;
  Angle targetAngle;
  Vector targetPosition,
         bestTargetPosition;
  //start calculations
  double minDist  =  1.0,
        maxDist  = 12.0,
        distStep =  1.0,
        bestValueForTargetPosition=0.0;
  TGLOGPOL(0, <<_2D<<VC2D(WSinfo::me->pos, maxDist, "ffff22"));
  Angle minAng   =  0.0,
        maxAng   = 2*PI,
        angStep  = DEG2RAD(15);
  Vector dummyPredIcpt, bestDummyPredIcpt;

  //HAUPTSCHLEIFE: ueber alle positionen, an denen ich mich anbieten kann
  for ( int i=0; i <= (maxDist-minDist)/distStep; i++)
  {
    double targetDistance = minDist + i*distStep;
    for (int j=0; j < (maxAng-minAng)/angStep; j++)
    {
      targetAngle = minAng + j*angStep;
      targetPosition.setX( targetDistance * cos(targetAngle) );
      targetPosition.setY( targetDistance * sin(targetAngle) );
      targetPosition += teammate->pos;

      bool debug = false; if (i==0 && j==0) debug = true;
      double valueForTargetPosition
        = evaluatePenaltyAreaOfferingPosition( teammate, 
                                               targetPosition, 
                                               debug );

      if (valueForTargetPosition < 0.0) continue; //position excluded
      
      //TGLOGPOL(0, "OFFPOSEVAL: " << targetPosition-teammate->pos << ": "<<((int)(valueForTargetPosition*1000))/10.0);
if ((float)rand()/(float)RAND_MAX < 0.0005) {
  TGLOGPOL(0, _2D << VSTRING2D( targetPosition,
                               ((int)(valueForTargetPosition*1000))/10.0,
                               "ffff66" ));
}
      if (valueForTargetPosition > bestValueForTargetPosition)
      {
        bestValueForTargetPosition = valueForTargetPosition;
        bestTargetPosition = targetPosition;
        /*bestV4TP_1 = valueForTargetPosition_1;
        bestV4TP_2 = valueForTargetPosition_2;
        bestV4TP_3 = valueForTargetPosition_3;
        bestV4TP_4 = valueForTargetPosition_4;
        bestV4TP_5 = valueForTargetPosition_5;*/
      }
    }
  }

  TGLOGPOL(0, "BEST PENALTYAREA-POS VALUE: "<<bestValueForTargetPosition);
  if (bestValueForTargetPosition >= 0.0)
  {
    offeringPosition = bestTargetPosition;
    TGLOGPOL(0, _2D << VC2D(offeringPosition, 1.0, "ffcc00"));
    TGLOGPOL(0, _2D << VSTRING2D( offeringPosition,
                                 ((int)(bestValueForTargetPosition*1000))/10.0,
                                 "ffcc00" ));
  }

  if (bestValueForTargetPosition <= 0.0)
  {
    bestValueForTargetPosition = 0.0;
    TGLOGPOL(0, << "OpponentAwarePositioning: WARNING: Did not find any offering situation!");
  }
  return bestValueForTargetPosition;
}


//============================================================================
// getBestAnticipatingOfferingPosition
//============================================================================
/**
 *  
 */
double
OpponentAwarePositioning::getBestAnticipatingOfferingPosition
                             ( PPlayer teammate,
                               Vector & offeringPosition,
                               PPlayer& passGivingTeammate )
{
  if (!teammate) return -1.0;
  int numberOfCriteria = MAX_CRITERIA;
#if LOGGING && BASIC_LOGGING
  double bestCriteriaValues[numberOfCriteria];
#endif
  double criteriaValues [numberOfCriteria];
  double valueForTargetPosition = 0.0;
  double bestValueForTargetPosition = -1.0;
  double targetDistance;
  Angle targetAngle;
  Vector targetPosition,
         bestTargetPosition;
  //value init
  for (int i=0; i<numberOfCriteria; i++) criteriaValues[i] = 0.0;
#if LOGGING && BASIC_LOGGING
  for (int i=0; i<numberOfCriteria; i++) bestCriteriaValues[i] = 0.0;
#endif
  //start calculations
  passGivingTeammate = NULL;
  Vector passStartPosition 
    = calculateExpectedPassStartPositionForAnticipatingOffering
      (teammate, passGivingTeammate, false);//no debug
  double minDist  =  2.0,
        maxDist  = 12.0,
        distStep =  1.0;
  if (passGivingTeammate)
  {
    maxDist = WSinfo::me->pos.distance(passGivingTeammate->pos);
    maxDist += passGivingTeammate->pos.distance(passStartPosition);
  }
  if (maxDist > 15.0) maxDist = 15.0;
  if (maxDist <  5.0) maxDist =  5.0;
  TGLOGPOL(0, <<_2D<<VC2D(WSinfo::me->pos, maxDist, "88ff88"));
  Angle minAng   =  0.0,
        maxAng   = 2*PI,
        angStep  = DEG2RAD(15);
Vector dummyPredIcpt, bestDummyPredIcpt;

  //HAUPTSCHLEIFE: ueber alle positionen, an denen ich mich anbieten kann
  for ( int i=0; i <= (maxDist-minDist)/distStep; i++)
  {
    targetDistance = minDist + i*distStep;
    for (int j=0; j < (maxAng-minAng)/angStep; j++)
    {
      targetAngle = minAng + j*angStep;
      targetPosition.setX( targetDistance * cos(targetAngle) );
      targetPosition.setY( targetDistance * sin(targetAngle) );
      targetPosition += teammate->pos;

      if (   OpponentAwarePositioning::shallIExcludeOfferPositionFromConsideration
                                       (teammate, targetPosition, 
                                        passStartPosition, passGivingTeammate)
          ||
             ( passGivingTeammate && targetPosition.getX() < passGivingTeammate->pos.getX()) /*TG06*/
         )
      {
        if ((float)rand()/(float)RAND_MAX < 0.005) {
          TGLOGPOL(0, _2D << VSTRING2D( targetPosition,
                                       "x",
                                       "ffff66" ));
        }
        continue;
      }
      //main function call: we evaluate a single position
      valueForTargetPosition = evaluateAnticipatingOfferingPosition( teammate,
                                                                     targetPosition,
                                                                     criteriaValues,
                                                                     passStartPosition);
      if (valueForTargetPosition < 0.0) continue; //position excluded
      
      //TGLOGPOL(0, "OFFPOSEVAL: " << targetPosition-teammate->pos << ": "<<((int)(valueForTargetPosition*1000))/10.0);
if ((float)rand()/(float)RAND_MAX < 0.005) {
  TGLOGPOL(0, _2D << VSTRING2D( targetPosition,
                               ((int)(valueForTargetPosition*1000))/10.0,
                               "ffff66" ));
}
      if (valueForTargetPosition > bestValueForTargetPosition)
      {
        bestValueForTargetPosition = valueForTargetPosition;
        bestTargetPosition = targetPosition;
#if LOGGING && BASIC_LOGGING
        for (int k=0; k<numberOfCriteria; k++)
          bestCriteriaValues[k] = criteriaValues[k];
#endif
      }
    }
  }

  if (bestValueForTargetPosition >= 0.0)
  {
  offeringPosition = bestTargetPosition;
  TGLOGPOL(0, _2D << VC2D(offeringPosition, 1.0, "ffcc00"));
  TGLOGPOL(0, _2D << VSTRING2D( offeringPosition,
                               ((int)(bestValueForTargetPosition*1000))/10.0,
                               "ffcc00" ));
for (int k = 0; k < numberOfCriteria; ++k) 
{
  TGLOGPOL(0, "BEST ANTICIPATING OFFPOS: criteria["<<k<<"] = "<<bestCriteriaValues[k] << "  w="
                       <<cvCriteriaWeights[ getRole(teammate->number) ][k]
                       <<"  -> "<<bestCriteriaValues[k]*cvCriteriaWeights[ getRole(teammate->number) ][k]);
}
double dummyFreeness;
OpponentAwarePositioning::isPassWayFreeEnoughForThisPosition
  ( teammate,
    bestTargetPosition,//offer pos
    passStartPosition,//pass start pos
    dummyFreeness,
    true,//debug
    false,//penalty area
    true);//handed over pass start pos
TGLOGPOL(0, "    ==> "<< bestValueForTargetPosition<<" at "<<offeringPosition<<" (pass from "<<passStartPosition<<")");
TGLOGPOL(0, <<_2D<<VL2D(passStartPosition,
                       offeringPosition, "ffff66"));
  }
  if (bestValueForTargetPosition <= 0.0)
  {
    bestValueForTargetPosition = 0.0;
    TGLOGPOL(0, << "OpponentAwarePositioning: WARNING: Did not find any ANTICIPATING offering situation!");
  }
  return bestValueForTargetPosition;
}


//============================================================================
// evaluateOfferingPosition
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::evaluateOfferingPosition
  ( PPlayer teammate,
    Vector  &offeringPosition,
    double  *criteriaValues )
{
  
  double * criteriaWeights = cvCriteriaWeights[getRole(teammate->number)];
  
  Vector passStartPos;
  isPassWayFreeEnoughForThisPosition( teammate,
                                      offeringPosition,
                                      passStartPos,
                                      criteriaValues[4], //freenessScore,
                                      false);
  criteriaValues[0] = evaluateOfferingPositionRegardingXPosition
                      (teammate, offeringPosition, passStartPos );
  criteriaValues[1] = evaluateOfferingPositionRegardingBallDistance
                      (teammate, offeringPosition, passStartPos );
  criteriaValues[2] = evaluateOfferingPositionRegardingDist2DirectOpp
                      (teammate, offeringPosition);
  criteriaValues[3] = evaluateOfferingPositionRegardingStamina
                      (teammate, offeringPosition);
  
  double valueSum = 0.0, weightSum = 0.0;
  for (int k=0; k<MAX_CRITERIA; k++)
  {
    valueSum += criteriaValues[k] * criteriaWeights[k];
    weightSum += criteriaWeights[k];
  }
  double valueForTargetPosition = valueSum / weightSum;
  
  return valueForTargetPosition;
}

//============================================================================
// evaluateAnticipatingOfferingPosition
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::evaluateAnticipatingOfferingPosition
  ( PPlayer teammate,
    Vector  &offeringPosition,
    double  *criteriaValues,
    Vector  passStartPos )
{
  
  double * criteriaWeights = cvCriteriaWeights[getRole(teammate->number)];
  
  isPassWayFreeEnoughForThisPosition( teammate,
                                      offeringPosition,
                                      passStartPos,
                                      criteriaValues[4], //freenessScore,
                                      false,  //debug
                                      false,  //considerationForPenaltyArea
                                      true);  //useHandedOverPassStartPosition

  criteriaValues[0] = evaluateOfferingPositionRegardingXPosition
                      (teammate, offeringPosition, passStartPos );
  criteriaValues[1] = evaluateOfferingPositionRegardingBallDistance
                      (teammate, offeringPosition, passStartPos );
  criteriaValues[2] = evaluateOfferingPositionRegardingDist2DirectOpp
                      (teammate, offeringPosition);
  criteriaValues[3] = evaluateOfferingPositionRegardingStamina
                      (teammate, offeringPosition);
  
  double valueSum = 0.0, weightSum = 0.0;
  for (int k=0; k<MAX_CRITERIA; k++)
  {
    valueSum += criteriaValues[k] * criteriaWeights[k];
    weightSum += criteriaWeights[k];
  }
  double valueForTargetPosition = valueSum / weightSum;
  
  return valueForTargetPosition;
}

//============================================================================
// K1: evaluateOfferingPositionRegardingXPosition
//============================================================================
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingXPosition
  ( PPlayer  teammate,
    Vector & offeringPosition,
    Vector   passStartPos )
{
  double maxDist = 15.0;/*TG=&: old: 10.0*/

  double xDelta = offeringPosition.getX() - passStartPos.getX();
  if (xDelta >  maxDist) xDelta =  maxDist;
  if (xDelta < -maxDist) xDelta = -maxDist;
  double xEval = ((xDelta) / (2.0*maxDist)) + 0.5; //yields value in [0;1.0]
  if (offeringPosition.getX() > FIELD_BORDER_X-10.0) xEval = 0.5;
  if (offeringPosition.getX() > FIELD_BORDER_X- 1.0) xEval = 0.0;

  double yDelta = offeringPosition.getY() - passStartPos.getY(),
        yEval;
  if (yDelta >  maxDist) yDelta =  maxDist;
  if (yDelta < -maxDist) yDelta = -maxDist;
  switch (WSinfo::me->number)
  {
    case 2: case 6: case 9: //left wing
    {
      yEval = (yDelta / (2.0*maxDist)) + 0.5; 
      break;
    }
    case 3: case 7: case 10: //centers
    {
      yEval = (fabs(yDelta) / (maxDist)); 
      break;
    }
    case 5: case 8: case 11: //right wing
    {
      yEval = ((-yDelta) / (2.0*maxDist)) + 0.5; 
      break;
    }
    default: 
    { yEval = 0.5; }
  }
  if (fabs(offeringPosition.getY()) > FIELD_BORDER_Y-1.0) yEval = 0.0;

  //TG07: Mehr Fluegelspiel erwuenscht, daher Veraenderung von
  //      0.6:0.4 auf 0.5:0.5
  return (0.5 * xEval + 0.5 * yEval);
}

//============================================================================
// K2: evaluateOfferingPositionRegardingBallDistance
//============================================================================
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingBallDistance
  ( PPlayer  teammate,
    Vector & offeringPosition,
    Vector   passStartPos, 
    bool     weAreInPenaltyArea )
{
  double distToBall
    = (passStartPos - offeringPosition).norm();
  double minOptimalDistToBall = 11.0,  //... tbd ... MR fragen!
        maxOptimalDistToBall = 14.0,
        maxDistToBall        = 20.0;
  if (weAreInPenaltyArea)
  {
    minOptimalDistToBall =  8.0; // TG17: 8->4
    maxOptimalDistToBall = 12.0; // TG17: 12->8
    maxDistToBall        = 17.0; // TG17: 17->15

    if (OpponentAwarePositioning::cvUseVersion2017)
    {
    minOptimalDistToBall =  4.0; // TG17: 8->4
    maxOptimalDistToBall = 8.0; // TG17: 12->8
    maxDistToBall        = 15.0; // TG17: 17->15
    }
  }
  if (distToBall < minOptimalDistToBall) //i.e. d2b in [0..7]
    return pow(distToBall / minOptimalDistToBall, 2.0);
  if (distToBall < maxOptimalDistToBall) //i.e. d2b in [7..12]
    return 1.0;
  if (distToBall < maxDistToBall) //i.e. d2b in [7..20]
    return (maxDistToBall-distToBall)/(maxDistToBall-maxOptimalDistToBall);
  //otherwise: d2b > 20  
  return 0.0;
}

//============================================================================
// K3: evaluateOfferingPositionRegardingDist2DirectOpp
//============================================================================
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingDist2DirectOpp
  ( PPlayer teammate,
    Vector & offeringPosition )
{
  PPlayer directOpponent;
  if ( cvFormation.getDirectOpponent(teammate->number, directOpponent) )
  {
    double distToDirectOpponent
      = (offeringPosition - directOpponent->pos).norm();
    double maxDistToDirectOpponent = 10.0;
    if (distToDirectOpponent > maxDistToDirectOpponent) 
      distToDirectOpponent = maxDistToDirectOpponent;
    return 1.0 - (distToDirectOpponent / maxDistToDirectOpponent);
  }
  else
  {
    //no direct opponent found
    return 1.0;
  }
}



//============================================================================
// K4: evaluateOfferingPositionRegardingStamina
//============================================================================
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingStamina
  ( PPlayer teammate,
    Vector & offeringPosition )
{
  double distanceToGo = (teammate->pos+teammate->vel - offeringPosition).norm();
  double maxDistanceToGo = 10.0;
  return 1.0 - (distanceToGo / maxDistanceToGo);
}

//============================================================================
// evaluateOfferingPositionRegardingDistanceToOpponents
//============================================================================
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingDistanceToOpponents
  ( PPlayer teammate,
    Vector & offeringPosition )
{
  PlayerSet opponents = WSinfo::valid_opponents;
  double maxDist = WSinfo::me->pos.distance(offeringPosition);
  opponents.keep_players_in_circle( offeringPosition, 
                                    maxDist);
  double returnValue = 1.0;
  double standardOpponentImpairment = 0.5;
  for (int i=0; i<opponents.num; i++)
  {
    double oppImpairment
      =   standardOpponentImpairment * sqrt( 
             (1.0 - opponents[i]->pos.distance(offeringPosition) / maxDist) );
    returnValue -= oppImpairment;
  }
  if (returnValue < 0.0) returnValue = 0.0;

  return returnValue;
}

//============================================================================
// evaluateOfferingPositionRegardingDistanceFromPenaltyAreaHomePos
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::evaluateOfferingPositionRegardingDistanceFromPenaltyAreaHomePos
  ( PPlayer teammate,
    Vector & offeringPosition )
{
  Vector myPenaltyAreaHomePos = getPenaltyAreaOfferingHomePosition(teammate->number);
  double dist = offeringPosition.distance(myPenaltyAreaHomePos),
        maxDist = 30.0;
  return Tools::max(0.0, 1.0 - (dist/maxDist));
}

//============================================================================
// shallIExcludeOfferPositionFromConsideration
//============================================================================
/**
 * This method is used to exlcude certain offering positions a priori.
 * 
 */
bool
OpponentAwarePositioning::shallIExcludeOfferPositionFromConsideration
                                    (PPlayer teammate,
                                     Vector  targetPosition,
                                     Vector  passStartPosition,
                                     PPlayer passStartPlayer)
{
  if (fabs(targetPosition.getX()) > FIELD_BORDER_X - 2.5)
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": near x border");
    return true;
  }
  
  if (fabs(targetPosition.getY()) > FIELD_BORDER_Y - 2.0)
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": near y border");
    return true;
  }


  if (   (    targetPosition.getX() >   WSinfo::his_team_pos_of_offside_line()
                                + WSmemory::get_his_offsideline_movement() -0.5 /*TG06:- 0.5*/
           && passStartPlayer == NULL )
       ||
         (    passStartPlayer != NULL
           && targetPosition.getX() > passStartPosition.getX()
           && passStartPosition.getX() >  WSinfo::his_team_pos_of_offside_line()
                                   + WSmemory::get_his_offsideline_movement() -0.5 )
       ||
         (    passStartPlayer != NULL
           && targetPosition.getX() >    WSinfo::his_team_pos_of_offside_line()
                                  + WSmemory::get_his_offsideline_movement()
           && passStartPosition.getX() <=  WSinfo::his_team_pos_of_offside_line()
                                    + WSmemory::get_his_offsideline_movement() -0.5)
     )
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": near his offside line (passStPl="
      <<(passStartPlayer==NULL?"NULL":"NOTnull")<<", tgtPos="
      <<targetPosition<<", passStPos="<<passStartPosition<<", hisOL="
      <<WSinfo::his_team_pos_of_offside_line()<<", olMove="
      <<WSmemory::get_his_offsideline_movement()<<")");
    return true;
  }

  if ( targetPosition.getX() < WSinfo::my_team_pos_of_offside_line() - 0.5 )
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": near my offside line");
    return true;
  }

  double maximalBackwardDeltaAtX = WSinfo::his_team_pos_of_offside_line(),
        minimalBackwardDeltaAtX = WSinfo::my_team_pos_of_offside_line(),
        maximalBackwardDelta    = -10.0,
        minimalBackwardDelta    =   1.0,
        backwardDelta = (  (WSinfo::me->pos.getX() - minimalBackwardDeltaAtX)
                         / (maximalBackwardDeltaAtX - minimalBackwardDeltaAtX))
                        * (maximalBackwardDelta - minimalBackwardDelta)
                        + minimalBackwardDelta; 
                        
  if (    passStartPlayer != NULL //parameter are given
       && WSinfo::ball->pos.getX() - passStartPosition.getX() > 7.5
       && WSinfo::me->pos.getX() < 10.0  )
    backwardDelta = minimalBackwardDelta;
  if (    getRole(WSinfo::me->number) == PT_ATTACKER //attackers don't go too
       && WSinfo::me->pos.getX() < 30.0 )                 //much behind to offer
    backwardDelta = minimalBackwardDelta;
                        
  if (targetPosition.getX() - WSinfo::me->pos.getX() < backwardDelta)
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": too much behind (backwardDelta="<<backwardDelta<<")");
    return true;
  }

  PPlayer nextBallPossessingTeammate = passStartPlayer;
  Vector expectedPassStartPos = passStartPosition;
  if (nextBallPossessingTeammate == NULL)
  {
    expectedPassStartPos
      = calculateExpectedPassStartPosition( nextBallPossessingTeammate );
  }
  if (nextBallPossessingTeammate)
  {
    double expectedPassLength = (expectedPassStartPos - targetPosition).norm(),
          myDistanceToGoForOffering = (WSinfo::me->pos - targetPosition).norm();
    if (expectedPassLength < myDistanceToGoForOffering )
    {
      TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": this position is silly, because the resulting pass would be shorter than the way i have to go");
      return true;
    }
    if (expectedPassStartPos.distance(targetPosition) < 6.0)
    {
        TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": this position is silly, because the resulting pass would be very short (only "<<expectedPassStartPos.distance(targetPosition)<<"m)");
        return true;
    }
    ANGLE potentialPassAngle = (targetPosition - expectedPassStartPos).ARG();
    if (    fabs(potentialPassAngle.get_value_mPI_pPI()) > (115.0/180.0)*PI 
         &&   targetPosition.distance(expectedPassStartPos)
            > WSinfo::me->pos.distance(expectedPassStartPos)
         && targetPosition.getX() < WSinfo::my_team_pos_of_offside_line() + 5.0 )
    {
        TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": this position is silly, "
          <<"because it is too unlikely that a backpass will be played when being "
          <<"still so far from his goal.");
        return true;
    }
    if (    nextBallPossessingTeammate->vel.getX() >= 0.0
         && WSinfo::ball->vel.getX() >= 0.0
         &&   fabs((expectedPassStartPos-WSinfo::me->pos).ARG().get_value_mPI_pPI()) 
            < PI/6.0
         && expectedPassLength > 8.0
       )
    {
      TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": too extreme back pass");
      return true;
    }
  }


  Vector predictedBallInterceptionPosition;
  double dummy;
  if ( ! OpponentAwarePositioning::isPassWayFreeEnoughForThisPosition
                                   (teammate,
                                    targetPosition,
                                    expectedPassStartPos,//ref
                                    dummy, //frenessscore
                                    false, //debug
                                    false, //passStartPositionIsBallPosition
                                    true ) //useHandedOverPassStartPosition
     )
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": passway not free, "<<dummy);
    return true;
  }

  Quadrangle2d  runWay( teammate->pos, targetPosition, 4.0, 4.0 );
  if (cvInterceptInformation.teammateStepsToGo > 0)
  {
    if ( runWay.inside(WSinfo::ball->pos+WSinfo::ball->vel) ) 
    {
      TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": my runway crowded, ball in runway");
      return true;
    }
    if ( runWay.inside( predictedBallInterceptionPosition ) ) 
    {
      TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": my runway crowded, ball icpt pos in runway");
      return true;
    }
  }

  PlayerSet teammatesOnMyRunWay = WSinfo::valid_teammates;
  teammatesOnMyRunWay.remove( teammate );
  teammatesOnMyRunWay.keep_players_in( runWay );
  if ( teammatesOnMyRunWay.num > 1 ) 
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": my runway crowded, teammates on runway");
    return true;
  }
  
  if ( expectedPassStartPos.distance( targetPosition ) < 5.0 )
  {
    TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": target too near to expected pass start pos");
    return true;
  }
    
  for (int i=0; i<WSinfo::valid_teammates.num; i++)
  {
    PPlayer p = WSinfo::valid_teammates[i];
    if (   p != teammate
        && p->pos.distance( WSinfo::ball->pos ) < 20.0
        &&    p->pos.distance(targetPosition) 
           < teammate->pos.distance(targetPosition) * 1.2 )
    {
      TGLOGPOL(4, "EXCLUDE pos "<<targetPosition<<": teammate can reach target faster");
      return true;
    }
  }
    
  return false;
}

//============================================================================
// calculateExpectedPassStartPositionForPenaltyArea
//============================================================================
Vector
OpponentAwarePositioning::
calculateExpectedPassStartPositionForPenaltyArea
  ( PPlayer consideredTeammate, PPlayer & passPlayingTeammate, bool debug )
{
  //compute the point of ball interception -> from where a pass may be played
  PPlayer  nextBallPossessingTeammate 
    = WSinfo::get_teammate_by_number(cvInterceptInformation.teammateNumber);
  if (!nextBallPossessingTeammate)  
  {
    TGLOGPOL(0,"OpponentAwarePositioning: "
      <<"calculateExpectedPassStartPositionForPenaltyArea: No next ball"
      <<" possessing teammate found (cvIcptInf.tmmNr="
      <<cvInterceptInformation.teammateNumber<<").");
    passPlayingTeammate = NULL;
    return Vector (0,0);
  }
    
  Vector predictedBallInterceptionPosition = WSinfo::ball->pos;
  Vector currentBallVelocity = WSinfo::ball->vel;
  if (cvInterceptInformation.teammateStepsToGo > 0)
  {
    int cnt = cvInterceptInformation.teammateStepsToGo;
    while (cnt > 0)
    {
      predictedBallInterceptionPosition += currentBallVelocity;
      currentBallVelocity *= ServerOptions::ball_decay;
      cnt--;
    }
  }
  if (predictedBallInterceptionPosition.getX() > FIELD_BORDER_X-1.0)
    predictedBallInterceptionPosition.setX( FIELD_BORDER_X-1.0 );
  if (predictedBallInterceptionPosition.getY() > FIELD_BORDER_Y-1.0)
    predictedBallInterceptionPosition.setY( FIELD_BORDER_Y-1.0 );
  if (predictedBallInterceptionPosition.getY() <-FIELD_BORDER_Y+1.0)
    predictedBallInterceptionPosition.setY( -FIELD_BORDER_Y+1.0 );

  //now determine the line from the predictedBallInterceptionPosition
  //to my point (consideredTeammate->pos) and if there is someone in between
  PlayerSet potentialPassGivingPlayers = WSinfo::valid_teammates_without_me;
  potentialPassGivingPlayers.keep_players_in_quadrangle
    ( 
      predictedBallInterceptionPosition,
      consideredTeammate->pos,
      4.0,
      predictedBallInterceptionPosition.distance(consideredTeammate->pos)
    );
  if (debug) { TGLOGPOL(0,<<_2D<<Quadrangle2d
    ( 
      predictedBallInterceptionPosition,
      consideredTeammate->pos,
      4.0,
      predictedBallInterceptionPosition.distance(consideredTeammate->pos)
    ));
  }
  potentialPassGivingPlayers.keep_and_sort_closest_players_to_point
    (
      potentialPassGivingPlayers.num,
      consideredTeammate->pos
    );
    
  if (potentialPassGivingPlayers.num == 0)
  {
    passPlayingTeammate = nextBallPossessingTeammate;
    if (debug) {
      TGLOGPOL(0,<<_2D<<VL2D(predictedBallInterceptionPosition,consideredTeammate->pos,"8b0000"));
    }
    return predictedBallInterceptionPosition;
  }
  else
  {
    passPlayingTeammate = potentialPassGivingPlayers[0];
    for (int i=0; i<potentialPassGivingPlayers.num; i++)
    {
      if (debug) {
        TGLOGPOL(0,<<_2D<<VL2D(potentialPassGivingPlayers[i]->pos,consideredTeammate->pos,"8b0000"));
      }
    }
    for (int i=0; i<potentialPassGivingPlayers.num; i++)
    {
      PlayerSet hisPlayers = WSinfo::valid_opponents;
      hisPlayers.keep_players_in_quadrangle( predictedBallInterceptionPosition,
                                             potentialPassGivingPlayers[i]->pos, 
                                             2.0, 2.0 );
      if ( hisPlayers.num == 0 )
      {
        return potentialPassGivingPlayers[i]->pos;
      }
    }
    return predictedBallInterceptionPosition;
  }
}

//============================================================================
// calculateExpectedPassStartPositionForAnticipatingOffering
//============================================================================
/**
 * This is exactly the same as 
 * calculateExpectedPassStartPositionForPenaltyArea.
 */
Vector
OpponentAwarePositioning::
calculateExpectedPassStartPositionForAnticipatingOffering
  ( PPlayer consideredTeammate, PPlayer & passPlayingTeammate, bool debug )
{
  //compute the point of ball interception -> from where a pass may be played
  PPlayer  nextBallPossessingTeammate 
    = WSinfo::valid_teammates.get_player_by_number(cvInterceptInformation.teammateNumber);
  if (!nextBallPossessingTeammate)  
  {  
    passPlayingTeammate = NULL;
    return Vector (0,0);
  }
    
  Vector predictedBallInterceptionPosition = WSinfo::ball->pos;
  Vector currentBallVelocity = WSinfo::ball->vel;
  if (cvInterceptInformation.teammateStepsToGo > 0)
  {
    int cnt = cvInterceptInformation.teammateStepsToGo;
    while (cnt > 0)
    {
      predictedBallInterceptionPosition += currentBallVelocity;
      currentBallVelocity *= ServerOptions::ball_decay;
      cnt--;
    }
  }
  if (predictedBallInterceptionPosition.getX() > FIELD_BORDER_X-1.0)
    predictedBallInterceptionPosition.setX( FIELD_BORDER_X-1.0 );
  if (predictedBallInterceptionPosition.getY() > FIELD_BORDER_Y-1.0)
    predictedBallInterceptionPosition.setY( FIELD_BORDER_Y-1.0 );
  if (predictedBallInterceptionPosition.getY() <-FIELD_BORDER_Y+1.0)
    predictedBallInterceptionPosition.setY( -FIELD_BORDER_Y+1.0 );

  //now determine the line from the predictedBallInterceptionPosition
  //to my point (consideredTeammate->pos) and if there is someone in between
  PlayerSet potentialPassGivingPlayers = WSinfo::valid_teammates_without_me;
  potentialPassGivingPlayers.keep_players_in_quadrangle
    ( 
      predictedBallInterceptionPosition,
      consideredTeammate->pos,
      4.0,
      predictedBallInterceptionPosition.distance(consideredTeammate->pos)
    );
  if (debug) { TGLOGPOL(0,<<_2D<<Quadrangle2d
    ( 
      predictedBallInterceptionPosition,
      consideredTeammate->pos,
      4.0,
      predictedBallInterceptionPosition.distance(consideredTeammate->pos)
    ));
  }
  potentialPassGivingPlayers.keep_and_sort_closest_players_to_point
    (
      potentialPassGivingPlayers.num,
      consideredTeammate->pos
    );
    
  if (potentialPassGivingPlayers.num == 0)
  {
    passPlayingTeammate = nextBallPossessingTeammate;
    if (debug) {
      TGLOGPOL(0,<<_2D<<VL2D(predictedBallInterceptionPosition,consideredTeammate->pos,"8b0000"));
    }
    return predictedBallInterceptionPosition;
  }
  else
  {
    passPlayingTeammate = potentialPassGivingPlayers[0];
    for (int i=0; i<potentialPassGivingPlayers.num; i++)
    {
      if (debug) {
        TGLOGPOL(0,<<_2D<<VL2D(potentialPassGivingPlayers[i]->pos,consideredTeammate->pos,"8b0000"));
      }
    }
    TGLOGPOL(1,<<_2D<<VL2D(nextBallPossessingTeammate->pos,
      potentialPassGivingPlayers[0]->pos,"77ff77"));
    return potentialPassGivingPlayers[0]->pos;
  }
}

//============================================================================
// calculateExpectedPassStartPosition
//============================================================================
Vector
OpponentAwarePositioning::
calculateExpectedPassStartPosition( PPlayer & nextBallPossessingTeammate )
{
  //compute the point of ball interception -> from where a pass may be played
    nextBallPossessingTeammate 
      = WSinfo::get_teammate_by_number
                (cvInterceptInformation.teammateNumber);
    if (!nextBallPossessingTeammate)  return Vector (0,0);
    Vector predictedBallInterceptionPosition = WSinfo::ball->pos;
    Vector currentBallVelocity = WSinfo::ball->vel;
    if (cvInterceptInformation.teammateStepsToGo == 0)
    {
      PlayerSet nearOpps = WSinfo::valid_opponents;
      nearOpps.keep_and_sort_closest_players_to_point
               (1, nextBallPossessingTeammate->pos);
      double distToNearestOpp = 5.0; //default value, will be overridden
      if (nearOpps.num>0)
        distToNearestOpp 
          = (nearOpps[0]->pos - nextBallPossessingTeammate->pos).norm();
      //we assume that the ball possessing teammate will dribble along the
      //x axis
      Vector dribbleWay;
      if ( fabs(nextBallPossessingTeammate->ang.get_value_mPI_pPI()) < PI/3.0)
        dribbleWay.init_polar( 1.0, nextBallPossessingTeammate->ang );
      else
        dribbleWay.init_polar( 1.0, ANGLE(0.0) );
      if (nearOpps.num > 0)
        dribbleWay.normalize( 0.3 * distToNearestOpp );
      if (  WSinfo::his_team_pos_of_offside_line()
          - nextBallPossessingTeammate->pos.getX() < 10.0)
        dribbleWay.normalize(0.0);
      predictedBallInterceptionPosition = WSinfo::ball->pos + dribbleWay;
    }
    else
    {
      int cnt = cvInterceptInformation.teammateStepsToGo;
      while (cnt > 0)
      {
        predictedBallInterceptionPosition += currentBallVelocity;
        currentBallVelocity *= ServerOptions::ball_decay;
        cnt--;
      }
    }
    if (predictedBallInterceptionPosition.getX() > FIELD_BORDER_X-1.0)
      predictedBallInterceptionPosition.setX( FIELD_BORDER_X-1.0 );
    if (predictedBallInterceptionPosition.getY() > FIELD_BORDER_Y-1.0)
      predictedBallInterceptionPosition.setY( FIELD_BORDER_Y-1.0 );
    if (predictedBallInterceptionPosition.getY() <-FIELD_BORDER_Y+1.0)
      predictedBallInterceptionPosition.setY( -FIELD_BORDER_Y+1.0 );
    return predictedBallInterceptionPosition;
}

//============================================================================
// isPassWayFreeEnoughForThisPosition
//============================================================================
bool
OpponentAwarePositioning::isPassWayFreeEnoughForThisPosition
  ( PPlayer teammate,
    Vector & offeringPosition,
    Vector & passStartPosition,
    double  & freenessScore,
    bool debug,
    bool considerationForPenaltyArea,
    bool useHandedOverPassStartPosition)
{  
  //1. from where may a pass be played?
  PPlayer passGivingTeammate = NULL;
  if (useHandedOverPassStartPosition)
  {
    PlayerSet teammates = WSinfo::valid_teammates_without_me;
    teammates.keep_and_sort_closest_players_to_point(1, WSinfo::ball->pos);
    if (teammates.num > 0)
      passGivingTeammate = teammates[0];
  }
  else
  {
    passStartPosition 
      = calculateExpectedPassStartPosition(passGivingTeammate);
    if (considerationForPenaltyArea) //passStartPosition = WSinfo::ball->pos;
      passStartPosition
        = calculateExpectedPassStartPositionForPenaltyArea(
            teammate, passGivingTeammate, debug);
  }
  if (!passGivingTeammate)
  {
    TGLOGPOL(0,<<"WARNING: Could not determine the next ball possessing teammate.");
    return false;
  }
  
  //2. define the check are in which players are considered harmful
  //   and the corresponding player sets
    double distFromOffPosToPassStartPos
      = (offeringPosition - passStartPosition).norm();
    double assumedAveragePlayerVelocity = 0.7;
    double playerCheckCircleRadius = distFromOffPosToPassStartPos
                                    +   assumedAveragePlayerVelocity 
                                      * cvInterceptInformation.teammateStepsToGo;
    PlayerSet consideredTeammates = WSinfo::valid_teammates;
    consideredTeammates.remove( teammate );
    consideredTeammates.remove( passGivingTeammate );
    consideredTeammates.keep_players_in_circle( offeringPosition,
                                                playerCheckCircleRadius );
    PlayerSet consideredOpponents = WSinfo::valid_opponents;
    //consideredOpponents.keep_players_in_circle( offeringPosition,
    //                                            playerCheckCircleRadius );
    Vector extendedOfferingPosition = offeringPosition-passStartPosition;
    extendedOfferingPosition.normalize(distFromOffPosToPassStartPos*1.3);
    extendedOfferingPosition += passStartPosition;
    consideredOpponents.keep_players_in_quadrangle( passStartPosition,
                                                    extendedOfferingPosition,
                                                    4.0,//4.0,
                                                    4.0+distFromOffPosToPassStartPos*0.65 );//4.0+distFromOffPosToPassStartPos*0.65 );
if(debug) { TGLOGPOL(0,<<_2D<<Quadrangle2d( passStartPosition,
                                          extendedOfferingPosition,
                                          4.0,//4.0,
                                          4.0+distFromOffPosToPassStartPos*0.65));//4.0+distFromOffPosToPassStartPos*0.65));
}
  //3. compute danger scores for all _opponent_ players
    double opponentDanger[consideredOpponents.num];
    for (int i=0; i<consideredOpponents.num; i++)
      opponentDanger[i] = 
        OpponentAwarePositioning::computeOpponentDangerWhenOffering
                                  ( passStartPosition,
                                    cvInterceptInformation.teammateStepsToGo,
                                    offeringPosition,
                                    consideredOpponents[i],
                                    debug);
    double teammateDanger[consideredTeammates.num];
    for (int i = 0; i < consideredTeammates.num; ++i) 
      teammateDanger[i] =
        OpponentAwarePositioning::computeTeammateDangerWhenOffering
                                  ( passStartPosition,
                                    cvInterceptInformation.teammateStepsToGo,
                                    offeringPosition,
                                    consideredTeammates[i],
                                    debug );
  //4. compute the quality of the offering position
  double maxDanger = 3.0,
        currentDanger = 0.0,
        //we will double the allowed danger when getting nearer to opp goal
        extraDanger = (offeringPosition.getX()+FIELD_BORDER_X)/(2.0*FIELD_BORDER_X)*maxDanger;
  maxDanger += extraDanger;
  double maximalIndividualPlayerDanger = 0.0;

  for (int i=0; i<consideredOpponents.num; i++)
  {
if (debug) { TGLOGPOL(0,<<"oppDanger["<<i<<"] for opp "<<consideredOpponents[i]->number<<"="<<opponentDanger[i]); }
    currentDanger += opponentDanger[i]*opponentDanger[i];
    if (opponentDanger[i] > maximalIndividualPlayerDanger)
    {
      maximalIndividualPlayerDanger = opponentDanger[i];
    }
  }
  for (int i = 0; i < consideredTeammates.num; ++i) 
  {
if (debug) { TGLOGPOL(0,<<"tmmDanger["<<i<<"] for tmm "<<consideredTeammates[i]->number<<"="<<teammateDanger[i]); }
    currentDanger += 0.5 * teammateDanger[i]*teammateDanger[i];
    if (teammateDanger[i] > maximalIndividualPlayerDanger)
    {
      maximalIndividualPlayerDanger = teammateDanger[i];
    }
  }

if (debug) { TGLOGPOL(0, <<" => dangSum = "<<currentDanger<<" (freenessScore="<<freenessScore<<")"); }
  freenessScore = 1.0 - currentDanger / maxDanger;
  if (freenessScore < 0.0) freenessScore = 0.0;
  
  if (currentDanger < maxDanger)
    return true;
  else
    return false;
}

//============================================================================
// computeOpponentDangerWhenOffering
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::computeOpponentDangerWhenOffering
                          ( Vector  passStartPosition,
                            int     stepsToInterception,
                            Vector  offeringPosition,
                            PPlayer consideredOpponent,
                            bool debug )
{
  double dangerReferenceDistance = 0.0, actualDangerDistance;
  double minDangerReferenceDistance = 4.0;
  double maxDangerReferenceDistance
    = offeringPosition.distance( passStartPosition ) * 0.5 
             + minDangerReferenceDistance ;
  double assumedAveragePlayerVelocity = 0.7;
  minDangerReferenceDistance += stepsToInterception * assumedAveragePlayerVelocity;
  maxDangerReferenceDistance += stepsToInterception * assumedAveragePlayerVelocity;
  if (   consideredOpponent->pos.distance( passStartPosition )
       < offeringPosition.distance( passStartPosition ) )
  {
    //the opponent is nearer than me to the ball icpt position
    //Lotfuss calculation
    Vector lotToPassway = Tools::get_Lotfuss( passStartPosition,
                                              offeringPosition,
                                              consideredOpponent->pos );
    //if (debug) TGLOGPOL(0, <<_2D<<VL2D(consideredOpponent->pos,
    //                              lotToPassway, "6666ff"));
    //TGLOGPOL(0, "LOT: "<<passStartPosition<<" "<<offeringPosition<<" "<<consideredOpponent->pos<< " => "<<lotToPassway);
    double icptWayPartition =   lotToPassway.distance( passStartPosition )
                             / offeringPosition.distance( passStartPosition );
    dangerReferenceDistance 
      = minDangerReferenceDistance
        + icptWayPartition * (maxDangerReferenceDistance-minDangerReferenceDistance);
    actualDangerDistance = consideredOpponent->pos.distance( lotToPassway );
if (debug) { TGLOGPOL(0, "PARTITION: "<<icptWayPartition<<" dangRefDist: "<<dangerReferenceDistance<<" actDangDist: "<<actualDangerDistance); }
  }
  else
  {
    //i am nearer than the opponent to the ball icpt position
    dangerReferenceDistance += maxDangerReferenceDistance;
    //if (debug) TGLOGPOL(0, <<_2D<<VC2D(offeringPosition, dangerReferenceDistance, "6666ff"));
    actualDangerDistance = consideredOpponent->pos.distance( offeringPosition );
  }
  if ( actualDangerDistance < 0.01 ) actualDangerDistance = 0.01;
  double danger = dangerReferenceDistance / actualDangerDistance;
  if (danger <  0.0) danger =  0.0;
  if (danger > 10.0) danger = 10.0;
  return danger;
}

//============================================================================
// computeTeammateDangerWhenOffering
//============================================================================
/**
 * 
 */
double
OpponentAwarePositioning::computeTeammateDangerWhenOffering
                          ( Vector  passStartPosition,
                            int     stepsToInterception,
                            Vector  offeringPosition,
                            PPlayer consideredTeammate,
                            bool debug )
{
  double dangerReferenceDistance, actualDangerDistance;
  double minDangerReferenceDistance = 1.0;
  dangerReferenceDistance
    = offeringPosition.distance( passStartPosition ) * 0.5 
      + minDangerReferenceDistance;
  double assumedAveragePlayerVelocity = 0.7;
  dangerReferenceDistance += stepsToInterception * assumedAveragePlayerVelocity;

  //if (debug) TGLOGPOL(0, <<_2D<<VC2D(offeringPosition, dangerReferenceDistance, "6666ff"));

  actualDangerDistance = consideredTeammate->pos.distance( offeringPosition );
if (debug) { TGLOGPOL(0,<<consideredTeammate->number<<"->dist:"<<actualDangerDistance<<"  refDist:"<<dangerReferenceDistance); }
  if ( actualDangerDistance < 0.01 ) actualDangerDistance = 0.01;
  double danger = dangerReferenceDistance / actualDangerDistance;
  if (danger <  0.0) danger =  0.0;
  if (danger > 10.0) danger = 10.0;
  return danger;
}

//============================================================================
// getAttackerPositionInOpponentBackFour
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning::getAttackerPositionInOpponentBackFour(double xOffset)
{
  int myRole = getRole(WSinfo::me->number);
  if (myRole != PT_ATTACKER)
  {
    TGLOGPOL(0, <<"WARNING: This method should be invoked by attackers only (getAttackerPositionInOpponentBackFour()).");
  }
  
  PlayerSet hisBackFour = WSinfo::valid_opponents;
  PPlayer hisGoalie = WSinfo::get_opponent_by_number(WSinfo::ws->his_goalie_number);
  if (hisGoalie) hisBackFour.remove(hisGoalie);
  hisBackFour.keep_and_sort_players_by_x_from_right(4);
  double hisBackmostPlayerX;
  if (hisBackFour.num > 0)
    hisBackmostPlayerX = hisBackFour[0]->pos.getX();
  hisBackFour.keep_and_sort_players_by_y_from_left(4);
  if (hisBackFour.num == 4)
  {
    Vector targetPoints[5];
    Vector rechtesAus( hisBackFour[0]->pos.getX(), -FIELD_BORDER_Y ),
           linkesAus( hisBackFour[3]->pos.getX(), FIELD_BORDER_Y );
    for (int i=0; i<5; i++)
    {
      if (i==0)
        targetPoints[i] = rechtesAus + hisBackFour[i]->pos;
      else
      if (i==4)
        targetPoints[i] = hisBackFour[i-1]->pos + linkesAus;
      else
        targetPoints[i] = hisBackFour[i-1]->pos + hisBackFour[i]->pos;
      targetPoints[i] *= 0.5;
      targetPoints[i].setX( hisBackmostPlayerX + xOffset );
      //keep minimal distances between target points
      if (targetPoints[4].getY() - targetPoints[3].getY() < 7.0 )
        targetPoints[4].setY( targetPoints[3].getY() + 7.0 );
      if (targetPoints[4].getY() > FIELD_BORDER_Y - 2.0 )
        targetPoints[4].setY( FIELD_BORDER_Y - 2.0 );
      if (targetPoints[3].getY() - targetPoints[2].getY() < 7.0 )
        targetPoints[3].setY( targetPoints[2].getY() + 7.0 );
      if (targetPoints[2].getY() - targetPoints[1].getY() < 7.0)
        targetPoints[1].setY( targetPoints[2].getY() - 7.0 );
      if (targetPoints[1].getY() - targetPoints[0].getY() < 7.0)
        targetPoints[0].setY( targetPoints[1].getY() - 7.0 );
      if (targetPoints[0].getY() < -FIELD_BORDER_Y + 2.0)
        targetPoints[0].setY( -FIELD_BORDER_Y + 2.0 );
    }
    for (int i=0; i<4; i++)
    {TGLOGPOL(2,<<_2D<<VL2D(targetPoints[i],
                           targetPoints[i+1],"bbffbb"));}
    Vector targetPoint;
    int targetPointIndex = 0;
    
    bool centerAttack    = fabs(WSinfo::ball->pos.getY()) <= 12.5,
         leftWingAttack  = WSinfo::ball->pos.getY() > 12.5,
         rightWingAttack = WSinfo::ball->pos.getY() < -12.5;
    
    switch (WSinfo::me->number)
    {
      case  9: 
      {
        targetPointIndex = 3;
        if (centerAttack)    targetPointIndex = 4;
        if (leftWingAttack)  targetPointIndex = 4;
        if (rightWingAttack) targetPointIndex = 3;
        targetPoint = targetPoints[targetPointIndex]; 
        if (targetPoint.getY() < 0.0) targetPoint.setY( hisBackFour[3]->pos.getY() + 7.0 );
        break;
      }
      case 10: 
      {
        targetPointIndex = 2;
        if (centerAttack)    targetPointIndex = 2;
        if (leftWingAttack)  targetPointIndex = 3;
        if (rightWingAttack) targetPointIndex = 1;
        targetPoint = targetPoints[targetPointIndex]; 
        break;
      }
      case 11: 
      {
        targetPointIndex = 1;
        if (centerAttack)    targetPointIndex = 0;
        if (leftWingAttack)  targetPointIndex = 1;
        if (rightWingAttack) targetPointIndex = 0;
        targetPoint = targetPoints[targetPointIndex]; 
        if (targetPoint.getY() > 0.0) targetPoint.setY( hisBackFour[0]->pos.getY() - 7.0 );
        break;
      }
      default:
        targetPoint = WSinfo::me->pos;
        TGLOGPOL(0, <<"WARNING: This method finds only positions for number 9,10,11 (return WSinfo::me->pos).");
        return WSinfo::me->pos;
    }
    
    /*TG-BREMEN begin*/ //scoring area consideration
    if (    WSinfo::ball->pos.distance(HIS_GOAL_CENTER) < 22.0
         && WSinfo::me->number == 10  
         && targetPointIndex == 1 )
    {
      if (targetPoint.getY() > 8.0) targetPointIndex --;
      if (targetPoint.getY() <-8.0) targetPointIndex ++;
      targetPoint = targetPoints[targetPointIndex]; 
    }
    if ( targetPoint.distance(WSinfo::ball->pos) < 5.0 )
    {
      if (targetPointIndex == 0 || targetPointIndex == 1) targetPointIndex ++ ;
      if (targetPointIndex == 3 || targetPointIndex == 4) targetPointIndex -- ;
      if (targetPointIndex == 2)
      {
        if (  WSinfo::me->pos.distance( targetPoints[1] )
            < WSinfo::me->pos.distance( targetPoints[3] ))
          targetPointIndex = 1;
        else
          targetPointIndex = 3;
      }
      targetPoint = targetPoints[targetPointIndex]; 
    }
    /*TG-BREMEN end*/

    return targetPoint;
  }
  else
  {
    TGLOGPOL(0, <<"WARNING: Could not detect his back four (return WSinfo::me->pos).");
    return WSinfo::me->pos;
  }
}

//############################################################################
//NEW VARIABLES & METHODS 2007
//############################################################################
#if 0
#define   TG07LOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TG07LOGPOL(YYY,XXX)
#endif

double OpponentAwarePositioning::cvCurrentCriticalOffsideLine = 0.0;
double OpponentAwarePositioning::cvOffsideLineRelaxation      = 0.0;

//============================================================================
// getAttackScoringAreaDefaultPositionForRole
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning::getAttackScoringAreaDefaultPositionForRole
                          ( int    myCurrentRole, 
                            Vector ourTeamNextBallPossessingPoint,
                            int    roleOfNextBallPossessingTeammate,
                            double  criticalOffsideLine )
{
  if ( WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_ATHUMBOLDT )
  {
    cvOffsideLineRelaxation = -1.75;
    TG07LOGPOL(0,<<"OpponentAwarePositioning: We play against ATH!");
  }
  else 
    cvOffsideLineRelaxation = -0.25;
  
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Called "
    <<"getAttackScoringAreaDefaultPositionForRole with parameters: "
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint
    <<" roleOfNextBallPossessingTeammate="<<roleOfNextBallPossessingTeammate
    <<" criticalOffsideLine="<<criticalOffsideLine);
  cvCurrentCriticalOffsideLine = criticalOffsideLine;
  switch (roleOfNextBallPossessingTeammate)
  {
    case ATTACK_ROLE_LEFT_ATTACKER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    case ATTACK_ROLE_CENTER_ATTACKER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    case ATTACK_ROLE_RIGHT_ATTACKER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingRightAttacker
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    {
      returnValue
        = getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder
             (myCurrentRole, ourTeamNextBallPossessingPoint);
      break;
    }    
    default:
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" the role of the next ball possessing (intercepting) teammate "
        <<"(roleOfNextBallPossessingTeammate="
        <<roleOfNextBallPossessingTeammate<<").");
    }    
  }
  return returnValue;
}

//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: WARNING: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker)"
        <<" indicate that MY ROLE is LEFT ATTACKER, and that the intercepting"
        <<" player is also the LEFT ATTACKER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
      returnValue.setX( ourTeamNextBallPossessingPoint.getX() );
      if (cvCurrentCriticalOffsideLine > ourTeamNextBallPossessingPoint.getX())
        returnValue.setX( 0.5 * (   cvCurrentCriticalOffsideLine
                                + ourTeamNextBallPossessingPoint.getX() ) - 2.0 );
      if (returnValue.getX() > FIELD_BORDER_X - 3.0)
        returnValue.setX( FIELD_BORDER_X - 3.0 );
      if (  returnValue.getX()
          >   cvCurrentCriticalOffsideLine 
            - (3.0+cvOffsideLineRelaxation))
        returnValue.setX(   cvCurrentCriticalOffsideLine
                          - (3.0+cvOffsideLineRelaxation) );
      //Y POSITION
      returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.3 );
      if ( ourTeamNextBallPossessingPoint.getY() - returnValue.getY() < 7.0 )
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 7.0 );
      if ( ourTeamNextBallPossessingPoint.getY() - returnValue.getY() > 13.0 )//10.0
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 13.0 );  //10.0
//      if (    WSinfo::ws->play_mode != PM_PlayOn
  //         && returnValue.y > 10.0 )
    //    returnValue.y = 10.0;
      //crossing
      if (ourTeamNextBallPossessingPoint.getY() < -5.0)
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 8.0 );
      //special case: hopeless target position
      if ( returnValue.getY() < ourTeamNextBallPossessingPoint.getY() )
      {
        Vector testPoint = returnValue;  testPoint.subFromY( 2.0 );
        Quadrangle2d checkArea( ourTeamNextBallPossessingPoint,
                                testPoint, 4.0, 6.0 );
        PlayerSet dangerousOpps = WSinfo::valid_opponents;
        dangerousOpps.keep_players_in( checkArea );
        if (dangerousOpps.num > 0)
        {
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.25 );
          if ( ourTeamNextBallPossessingPoint.getY() - returnValue.getY() < 7.0 )
            returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 7.0 );
        }
      }
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
      returnValue.setX( ourTeamNextBallPossessingPoint.getX() );
      if (cvCurrentCriticalOffsideLine > ourTeamNextBallPossessingPoint.getX())
        returnValue.setX(   cvCurrentCriticalOffsideLine
                          - (3.0+cvOffsideLineRelaxation) );
      if (returnValue.getX() > FIELD_BORDER_X - 7.0)
        returnValue.setX( FIELD_BORDER_X - 7.0 );
      //Y POSITION
      returnValue.setY( -9.0 ); //-7.0;ZUI_RA
//ZUI_RA      if ( ourTeamNextBallPossessingPoint.y > 20.0 )
  //ZUI_RA      returnValue.y = -3.0;
      if ( ourTeamNextBallPossessingPoint.getY() - returnValue.getY() < 16.0 )
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 16.0 );
      //crossing
      if (ourTeamNextBallPossessingPoint.getY() < -12.0)
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 8.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the positions of the left/center attackers
      Vector LAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          ( ATTACK_ROLE_LEFT_ATTACKER, ourTeamNextBallPossessingPoint );
      Vector CAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          ( ATTACK_ROLE_CENTER_ATTACKER, ourTeamNextBallPossessingPoint );
      //X POSITION
      double midfieldOffset = 8.0;
      if (ourTeamNextBallPossessingPoint.getX() < FIELD_BORDER_X - 6.0)
        midfieldOffset = 0.175 * (0.5 * (LAposition.getX() + CAposition.getX()));
      returnValue.setX( (0.5 * (LAposition.getX() + CAposition.getX())) - midfieldOffset );
      //Y POSITION
      double minY = 10.0, maxY = 0.7*LAposition.getY() + 0.3*CAposition.getY(),
            consideredLAposY = Tools::max(0.0,LAposition.getY());
      if (maxY <= minY) returnValue.setY( minY );
      else
        returnValue.setY( (consideredLAposY/FIELD_BORDER_Y)*(maxY-minY)+minY );
      if (returnValue.getY() > FIELD_BORDER_Y - 5.0)
        returnValue.setY( FIELD_BORDER_Y - 5.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position of the left midfielder
      Vector LMposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          ( ATTACK_ROLE_LEFT_MIDFIELDER, ourTeamNextBallPossessingPoint );
      //X POSITION
      returnValue.setX( LMposition.getX() - 2.0 );
      //Y POSITION
      double minY = -3.0, maxY = LMposition.getY()-10.0,
            consideredBallIcptPosY = Tools::max(0.0,
                                            ourTeamNextBallPossessingPoint.getY());
      if (maxY <= minY) returnValue.setY( minY );
      else
        returnValue.setY( (consideredBallIcptPosY/FIELD_BORDER_Y)*(maxY-minY)+minY );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position of the center midfielder
      Vector CMposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          ( ATTACK_ROLE_CENTER_MIDFIELDER, ourTeamNextBallPossessingPoint );
      //X POSITION
      returnValue.setX( CMposition.getX() );
      //Y POSITION
      double minY = -10.0, maxY = CMposition.getY()-10.0,
            consideredBallIcptPosY = Tools::max(0.0,
                                            ourTeamNextBallPossessingPoint.getY());
      if (maxY <= minY) returnValue.setY( minY );
      else
        returnValue.setY( (consideredBallIcptPosY/FIELD_BORDER_Y)*(maxY-minY)+minY );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}
  
//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  Vector returnValue;
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
      returnValue.setX(   cvCurrentCriticalOffsideLine
                        - (3.0+cvOffsideLineRelaxation) );
      if (returnValue.getX() > FIELD_BORDER_X-5.0)
        returnValue.setX( FIELD_BORDER_X-5.0 );
      //Y POSITION
      double yShift = 8.0;
      if (ourTeamNextBallPossessingPoint.getY() < -7.0)
        yShift *= (ourTeamNextBallPossessingPoint.getY()/(-7.0));
      else
      if (ourTeamNextBallPossessingPoint.getY() >  7.0)
        yShift /= (ourTeamNextBallPossessingPoint.getY()/(7.0));
      returnValue.setY( ourTeamNextBallPossessingPoint.getY() + yShift );
      //crossing
      if (    ourTeamNextBallPossessingPoint.getY() > 18.0
//           &&   WSinfo::me->pos.distance(ourTeamNextBallPossessingPoint) 
  //            < 5.0 + cvInterceptInformation.teammateStepsToGo 
         )
      {
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.5 );
        if (WSinfo::me->pos.distance(ourTeamNextBallPossessingPoint) < 2.0)
        { //don't disturb ball holder
          returnValue.setX( ourTeamNextBallPossessingPoint.getX() - 3.0 );
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 1.0 );
        }
      } 
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: ERROR: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker)"
        <<" indicate that MY ROLE is CENTER ATTACKER, and that the intercepting"
        <<" player is also the CENTER ATTACKER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricBallInterception = ourTeamNextBallPossessingPoint;
      symmetricBallInterception.mulYby( -1.0 );
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
          (ATTACK_ROLE_LEFT_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the left attacker
      Vector LAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
                                   ( ATTACK_ROLE_LEFT_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
      returnValue.setX( LAposition.getX() * 0.85 );
      //Y POSITION
      returnValue.setY( LAposition.getY() );
      if (ourTeamNextBallPossessingPoint.getY() > 10.0)
        returnValue.setY(   0.7 * LAposition.getY()
                          + 0.3 * ourTeamNextBallPossessingPoint.getY() );
      else
      if (ourTeamNextBallPossessingPoint.getY() < -10.0)
        returnValue.subFromY( 5.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
      returnValue.setX( ourTeamNextBallPossessingPoint.getX() * 0.85 );
      //Y POSITION
      returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.4 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the left attacker
      Vector RAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
                                   ( ATTACK_ROLE_RIGHT_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
      returnValue.setX( RAposition.getX() * 0.85 );
      //Y POSITION
      returnValue.setY( RAposition.getY() );
      if (ourTeamNextBallPossessingPoint.getY() > 10.0)
        returnValue.addToY( 5.0 );
      else
      if (ourTeamNextBallPossessingPoint.getY() < -10.0)
        returnValue.setY( 0.7 * RAposition.getY()
                          + 0.3 * ourTeamNextBallPossessingPoint.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}
  
//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingRightAttacker
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingRightAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingRightAttacker"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  Vector symmetricBallInterception = ourTeamNextBallPossessingPoint;
  symmetricBallInterception.mulYby( -1.0 );
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          (ATTACK_ROLE_RIGHT_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          (ATTACK_ROLE_CENTER_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: ERROR: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingRightAttacker)"
        <<" indicate that MY ROLE is RIGHT ATTACKER, and that the intercepting"
        <<" player is also the RIGHT ATTACKER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          (ATTACK_ROLE_RIGHT_MIDFIELDER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          (ATTACK_ROLE_CENTER_MIDFIELDER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
          (ATTACK_ROLE_LEFT_MIDFIELDER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}
  
//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
        //knapp vor abseitslinie
        double minDelta = 0.5,
               maxDelta = 4.0+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        //ausweichend gegenueber dem ballbesitzenden mittelfeldspieler
        if (ourTeamNextBallPossessingPoint.getY() > 22.0)
        { //ausweichend nach innen weg
          returnValue.setY( 0.75 * ourTeamNextBallPossessingPoint.getY() );
        }
        else
        { //ausweichend nach aussen weg
          returnValue.setY(
                ourTeamNextBallPossessingPoint.getY()
              + 0.5*(FIELD_BORDER_Y - ourTeamNextBallPossessingPoint.getY()) );
        }
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the left attacker
      Vector LAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( ATTACK_ROLE_LEFT_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
        //knapp vor abseitslinie
        double minDelta = 0.5,
              maxDelta = 4.0+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        //positionierung abhaengig vom linken stuermer
        if (LAposition.getY() > ourTeamNextBallPossessingPoint.getY())
        {
          returnValue.setY( 0.6 * ourTeamNextBallPossessingPoint.getY() );
          if (returnValue.getY() > 10.0) returnValue.setY( 10.0 ); //trunkieren
        }
        else
        {
          returnValue.setY( LAposition.getY() - 10.0 );
        }
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the left attacker
      Vector CAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( ATTACK_ROLE_CENTER_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
        //knapp vor abseitslinie
        double minDelta = 0.5,
              maxDelta = 2.0+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        returnValue.setY( CAposition.getY() - 8.0 );
//        if (returnValue.y > -4.0) returnValue.y = -4.0; //trunkieren
if (returnValue.getY() > -8.0) returnValue.setY( -8.0 ); //trunkieren //ZUI
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: ERROR: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder)"
        <<" indicate that MY ROLE is LEFT MIDFIELDER, and that the intercepting"
        <<" player is also the LEFT MIDFIELDER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
        returnValue.setX(
              0.7 * ourTeamNextBallPossessingPoint.getX()
            + 0.3 * cvCurrentCriticalOffsideLine );
        if (ourTeamNextBallPossessingPoint.getX() > FIELD_BORDER_X - 13.0)
          returnValue.setX( FIELD_BORDER_X - 16.0 );
        if (cvCurrentCriticalOffsideLine - returnValue.getX() < 8.0)
          returnValue.setX( cvCurrentCriticalOffsideLine - 8.0 );
        if (returnValue.getX() > FIELD_BORDER_X - 13.0)
          returnValue.setX( FIELD_BORDER_X - 13.0 );
      //Y POSITION
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.5 );
        if (ourTeamNextBallPossessingPoint.getY() - returnValue.getY() < 9.0)
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 9.0 );
        //crossing
        if (ourTeamNextBallPossessingPoint.getY() < 0.0)
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 9.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the center midfielder
      Vector CMposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( ATTACK_ROLE_CENTER_MIDFIELDER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
        returnValue.setX(
              0.6 * ourTeamNextBallPossessingPoint.getX()
            + 0.4 * cvCurrentCriticalOffsideLine );
        if (cvCurrentCriticalOffsideLine - returnValue.getX() < 5.0)
          returnValue.setX( cvCurrentCriticalOffsideLine - 5.0 );
        if (returnValue.getX() > FIELD_BORDER_X - 13.0)
          returnValue.setX( FIELD_BORDER_X - 13.0 );
      //Y POSITION
        returnValue.setY( CMposition.getY() - 12.0 );
        //crossing center midfielder
        if (ourTeamNextBallPossessingPoint.getY() < 0.0)
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 9.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}
  
//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the center attacker
      Vector CAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( ATTACK_ROLE_CENTER_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
        double minDelta = 0.5,
              maxDelta = 3.5+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        returnValue.setY( CAposition.getY() + 8.0 );
        if (returnValue.getY() < 0.0) returnValue.setY( 0.0 );
if (returnValue.getY() < 9.0) returnValue.setY( 9.0 ); //ZUI_RA
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
        double minDelta = 0.5,
              maxDelta = 3.5+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        if ( fabs(ourTeamNextBallPossessingPoint.getY()) < 10.0 )
        {
          if (ourTeamNextBallPossessingPoint.getX() > FIELD_BORDER_X-7.0)
          {
            if (ourTeamNextBallPossessingPoint.getY() > 0.0)
              returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 7.0 );
            else
              returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 7.0 );
          }
          else
          {
            returnValue.setY( 0.0 );
          }
        }
        else
        {
          if (ourTeamNextBallPossessingPoint.getX() > FIELD_BORDER_X-7.0)
          {
            if (ourTeamNextBallPossessingPoint.getY() > 0.0)
              returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 10.0 );
            else
              returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 10.0 );
          }
          else
          {
            returnValue.setY( ourTeamNextBallPossessingPoint.getY() * 0.5 );
          }
        }
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the center attacker
      Vector CAposition 
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( ATTACK_ROLE_CENTER_ATTACKER,
                                     ourTeamNextBallPossessingPoint );
      //X POSITION
        double minDelta = 0.5,
              maxDelta = 3.5+cvOffsideLineRelaxation;
        double actualDelta
          = ( Tools::max(0.0,cvCurrentCriticalOffsideLine - 30.0) / 22.5 ) 
            * (maxDelta-minDelta) + minDelta;
        returnValue.setX( cvCurrentCriticalOffsideLine - actualDelta );
      //Y POSITION
        returnValue.setY( CAposition.getY() - 8.0 );
        if (returnValue.getY() > 0.0) returnValue.setY( 0.0 );
if (returnValue.getY() > -9.0) returnValue.setY( -9.0 ); //ZUI_RA
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
        returnValue.setX(
              0.7 * ourTeamNextBallPossessingPoint.getX()
            + 0.3 * cvCurrentCriticalOffsideLine );
        if (cvCurrentCriticalOffsideLine - returnValue.getX() < 6.0)
          returnValue.setX( cvCurrentCriticalOffsideLine - 6.0 );
        if (  cvCurrentCriticalOffsideLine - ourTeamNextBallPossessingPoint.getX()
            < 6.0 )
          returnValue.setX( cvCurrentCriticalOffsideLine - 9.0 );
      //Y POSITION
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 10.0 );
        //crossing
        if (   ourTeamNextBallPossessingPoint.getY() > 20.0
            || (   ourTeamNextBallPossessingPoint.getY() < 14.0
                && ourTeamNextBallPossessingPoint.getX() > FIELD_BORDER_X-15.0 ) )
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 10.0 );
        if (returnValue.getY() < 0.0) returnValue.setY( 0.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: ERROR: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder)"
        <<" indicate that MY ROLE is CENTER MIDFIELDER, and that the intercepting"
        <<" player is also the CENTER MIDFIELDER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //X POSITION
        returnValue.setX(
              0.7 * ourTeamNextBallPossessingPoint.getX()
            + 0.3 * cvCurrentCriticalOffsideLine );
        if (cvCurrentCriticalOffsideLine - returnValue.getX() < 6.0)
          returnValue.setX( cvCurrentCriticalOffsideLine - 6.0 );
        if (  cvCurrentCriticalOffsideLine - ourTeamNextBallPossessingPoint.getX()
            < 6.0 )
          returnValue.setX( cvCurrentCriticalOffsideLine - 9.0 );
      //Y POSITION
        returnValue.setY( ourTeamNextBallPossessingPoint.getY() - 10.0 );
        //crossing
        if (   ourTeamNextBallPossessingPoint.getY() < -20.0
            || (   ourTeamNextBallPossessingPoint.getY() < -14.0
                && ourTeamNextBallPossessingPoint.getX() > FIELD_BORDER_X-15.0 ) )
          returnValue.setY( ourTeamNextBallPossessingPoint.getY() + 10.0 );
        if (returnValue.getY() > 0.0) returnValue.setY( 0.0 );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}
  
//============================================================================
// getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder
//============================================================================
/**
 * 
 */
Vector
OpponentAwarePositioning
  ::getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint )
{
  Vector returnValue;
  TG07LOGPOL(0,<<"OpponentAwarePositioning: Method getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder"
    <<" has been called with parameters"
    <<" myCurrentRole="<<myCurrentRole
    <<" ourTeamNextBallPossessingPoint="<<ourTeamNextBallPossessingPoint);
  Vector symmetricBallInterception = ourTeamNextBallPossessingPoint;
  symmetricBallInterception.mulYby( -1.0 );
  switch (myCurrentRole)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
          (ATTACK_ROLE_RIGHT_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
          (ATTACK_ROLE_CENTER_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_ATTACKER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
          (ATTACK_ROLE_LEFT_ATTACKER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_LEFT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
          (ATTACK_ROLE_RIGHT_MIDFIELDER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_CENTER_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      //Pre-determine the position for the symmetric situation
      Vector symmetricPosition
        = getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
          (ATTACK_ROLE_CENTER_MIDFIELDER,symmetricBallInterception);  
      //X POSITION
      returnValue.setX(  symmetricPosition.getX() );
      //Y POSITION
      returnValue.setY( -symmetricPosition.getY() );
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case ATTACK_ROLE_RIGHT_MIDFIELDER:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: ERROR: Parameters of this method"
        <<" (getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder)"
        <<" indicate that MY ROLE is RIGHT MIDFIELDER, and that the intercepting"
        <<" player is also the RIGHT MIDFIELDER. Return the value of "
        <<" ourTeamNextBallPossessingPoint.");
      returnValue = ourTeamNextBallPossessingPoint;
      break;
    }    
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    default:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      TG07LOGPOL(0,"OpponentAwarePositioning: SEVERE ERROR: I do not know"
        <<" my own role "<<"(myCurrentRole="
        <<myCurrentRole<<").");
    }    
  }
  return returnValue;
}



