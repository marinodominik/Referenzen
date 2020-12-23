#include "ws_memory.h"
#include "tools.h"
#include "ws_info.h"
#include "log_macros.h"
#include "intention.h"
#include "blackboard.h"
#include "../behaviors/skills/selfpass2_bms.h"
#include "Cmd.h"

#define MAX_VISIBLE_DISTANCE 50.0 // ignore objects further away (last_seen_to_point())
#define SELFPASS_FAILURE  0
#define SELFPASS_DONTCARE 1
#define SELFPASS_SUCCESS  2

WSmemory::ViewInfo WSmemory::view_info[MAX_VIEW_INFO];
WSmemory::AttentionToInfo WSmemory::attentionto_info[MAX_ATTENTIONTO_INFO];
InterceptResult WSmemory::cvCurrentInterceptResult[MAX_INTERCEPT_INFO_PLAYERS];
PlayerSet          WSmemory::cvCurrentInterceptPeople;

long   WSmemory::ball_was_kickable4me_at;

int    WSmemory::counter;
float  WSmemory::momentum[MAX_MOMENTUM];
int    WSmemory::opponent_last_at_ball_number;
long   WSmemory::opponent_last_at_ball_time = 0;
int    WSmemory::teammate_last_at_ball_number; // include myself!
long   WSmemory::teammate_last_at_ball_time = 0;
int    WSmemory::saved_team_last_at_ball;
int    WSmemory::saved_team_in_attack; //gibt an, welches Team im Angriff ist
double WSmemory::his_offside_line_lag2;
double WSmemory::his_offside_line_lag1;
double WSmemory::his_offside_line_lag0;
int    WSmemory::last_update_at;
long   WSmemory::last_kick_off_at = 0;
int    WSmemory::cvTimesHisGoalieSeen = 0;
int    WSmemory::cvTimesHisGoalieOutsidePenaltyArea = 0;
long   WSmemory::cvOurLastSelfPass                         = - 1;
long   WSmemory::cvOurLastSelfPassWithBallLeavingKickRange = - 1;
int    WSmemory::cvOurLastSelfPassRiskLevel                = 0;
int    WSmemory::cvSelfpassAssessment[MAX_SELFPASS_RISK_LEVELS][3];
int    WSmemory::cvLastTimeFouled = -1000;
int    WSmemory::cvLastFouledOpponentNumber = 0;
int    WSmemory::cvLastHisGoalKick = -1000;

void WSmemory::init() {
  for(int i=0;i<MAX_VIEW_INFO;i++) {
    view_info[i].time=-1;
  }
  for (int i=0; i<MAX_ATTENTIONTO_INFO; i++)
    attentionto_info[i].time = -1;

  ball_was_kickable4me_at = -1000; // init
  his_offside_line_lag2 = 0;
  his_offside_line_lag1 = 0;
  his_offside_line_lag0 = 0;
  last_update_at = -1;
  
  for (int i=0; i<MAX_SELFPASS_RISK_LEVELS; i++)
    for (int j=0; j<3; j++)
      cvSelfpassAssessment[i][j] = 0;
}

void WSmemory::update_fastest_to_ball() 
{  
  //draw the ball!
  LOG_POL( 1, << _2D << VC2D(      WSinfo::ball->pos,               0.4, "ff8c00" ) << std::flush );
  LOG_POL( 1, << _2D << VC2D(      WSinfo::ball->pos,               0.3, "ff8c00" ) << std::flush );
  LOG_POL( 1, << _2D << VSTRING2D( WSinfo::ball->pos, WSinfo::ball->age, "ff8c00" ) << std::flush );


  int local_team_in_attack                = saved_team_in_attack, 
      local_teammate_last_at_ball_number  = teammate_last_at_ball_number,                              
      local_teammate_last_at_ball_time    = teammate_last_at_ball_time, 
      local_opponent_last_at_ball_number  = opponent_last_at_ball_number, 
      local_opponent_last_at_ball_time    = opponent_last_at_ball_time,
      local_team_last_at_ball             = saved_team_last_at_ball;
  

  //berechnet mit Gedaechtnis wer im Angriff ist (=schnellster Spieler zum Ball)
  float summe = 0.0;
  PlayerSet pset_tmp = WSinfo::valid_teammates;
  pset_tmp += WSinfo::valid_opponents;
  InterceptResult intercept_res[MAX_INTERCEPT_INFO_PLAYERS];
  pset_tmp.keep_and_sort_best_interceptors_with_intercept_behavior_to_WSinfoBallPos
           (MAX_INTERCEPT_INFO_PLAYERS, /*WSinfo::ball->pos, WSinfo::ball->vel,*/ intercept_res);
  cvCurrentInterceptPeople = pset_tmp;
  for (int i=0; i<MAX_INTERCEPT_INFO_PLAYERS; i++) cvCurrentInterceptResult[i] = intercept_res[i];
  Vector ballIcptPosition = intercept_res[0].pos;
  LOG_POL(2,<<"Beste Interzeptoren:"<<std::flush);
  if (pset_tmp.num>0) { LOG_POL(2,<<"  1."<<pset_tmp[0]->team<<"  "<<pset_tmp[0]->number<<"  ("<<intercept_res[0].time<<")"); }
  if (pset_tmp.num>1) { LOG_POL(2,<<"  2."<<pset_tmp[1]->team<<"  "<<pset_tmp[1]->number<<"  ("<<intercept_res[1].time<<")"); }
  if (pset_tmp.num>2) { LOG_POL(2,<<"  3."<<pset_tmp[2]->team<<"  "<<pset_tmp[2]->number<<"  ("<<intercept_res[2].time<<")"); }
  int myFastestInterceptoAmongTopThree = -1;
  if (pset_tmp.num >= 1 && WSinfo::is_ball_pos_valid() ) 
  {
      //HIS_TEAM ist 2     ==> 0           ==> daher Addition von 1 und modulo 3
      //UNKNOWN_TEAM ist 0 ==> 1
      //MY_TEAM ist 1      ==> 2
      //==> guessedTeam: -1=HIS, 0=UNKNOWN, 1=MY [!!!]
      int guessedTeamInAttack = ( ( (pset_tmp[0]->team) + 1 ) % 3) - 1;
      if (pset_tmp[0]->team == MY_TEAM) myFastestInterceptoAmongTopThree = 0;
      else if (pset_tmp.num >= 2 && pset_tmp[1]->team == MY_TEAM) myFastestInterceptoAmongTopThree = 1;
      else if (pset_tmp.num >= 3 && pset_tmp[2]->team == MY_TEAM) myFastestInterceptoAmongTopThree = 2;
      if (    pset_tmp[0]->team == HIS_TEAM && intercept_res[0].pos.getX() > 15.0
           && myFastestInterceptoAmongTopThree > -1
           && (     fabs(  intercept_res[0].time
                         - intercept_res[myFastestInterceptoAmongTopThree].time )
                  < 0.25*intercept_res[0].time
               ||   intercept_res[myFastestInterceptoAmongTopThree].time 
                  - intercept_res[0].time <= 2) )
        guessedTeamInAttack = 0; //UNKNOWN, may be my team will intercept
      float w1 = (float)(MAX_MOMENTUM - WSinfo::ball->age) / (float)MAX_MOMENTUM;
      if (w1<0.0) w1 = 0.0;
      float w2 = (float)(MAX_MOMENTUM - pset_tmp[0]->age) / (float)MAX_MOMENTUM;
      if (w2<0.0) w2 = 0.0;
      if (w1+w2<0.0001) 
        momentum[counter] = 0.0;
      else
        momentum[counter] = ((w1+w2)*(float)guessedTeamInAttack) / (w1+w2);
    LOG_POL(2,<<"current momentum["<<counter<<"] = "<<momentum[counter]);
  } 
  else 
  {
      //leeres pset 
      momentum[counter] = 0.0;
  }
  LOG_POL(2,<<"ALL momentums:");
  for(int i=0; i<MAX_MOMENTUM;i++)
  {
      int index = (counter+MAX_MOMENTUM-i)%MAX_MOMENTUM;
      float w = (float)(MAX_MOMENTUM-i) / (float)MAX_MOMENTUM;
      LOG_POL(2,<<"    momentum["<<index<<"]="<<momentum[index]);
      summe += w * momentum[index];
  }
  LOG_POL(2,<<"SUMME = "<<summe);
  counter = (counter + 1 ) % 5;
  //Summe kann Werte zwischen -3 und 3 (==1+0.8+0.6+0.4+0.2) angenommen haben.
  //Folgende Zuordnung ist wuenschenswert ...
  //Summe in -3..0 ==> Gegner im Ballbesitz
  //         0..1 ==> unbekannt
  //         1..3==> wir im Ballbesitz
  float hisToUnknownThreshold = 0.0;
  float unknownToMyThreshold  = 0.9;
  //Je nachdem, wo sich das Spiel gerade abspielt, sollten diese 
  //Schwellwerte adaptiert werden!
  if (   (ballIcptPosition - MY_GOAL_CENTER).sqr_norm() < 20*20
      || (   ballIcptPosition.getX()
           - Tools::min( 0.0,
                         WSinfo::my_team_pos_of_offside_line()) < 7.5 ) )
  {
    hisToUnknownThreshold = 1.0;
    unknownToMyThreshold  = 1.9;
  }
  if (   (ballIcptPosition - MY_GOAL_CENTER).sqr_norm() > 60*60
      && (   ballIcptPosition.getX()
           - Tools::min( 0.0,
                         WSinfo::my_team_pos_of_offside_line()) > 15.0) )
  {
    hisToUnknownThreshold = -1.0;
    unknownToMyThreshold  = -0.1;
  }
  if (   
         myFastestInterceptoAmongTopThree > -1
      && 
         (   (    (WSinfo::ball->pos - HIS_GOAL_CENTER).sqr_norm() < 20*20
              && (ballIcptPosition - HIS_GOAL_CENTER).sqr_norm() < 35*35 )
          || ballIcptPosition.getX() > WSinfo::his_team_pos_of_offside_line() )
      )
  {
    hisToUnknownThreshold = -2.0;
    unknownToMyThreshold  = -1.1;
  }
  ///////ENDE DER SCHWELLWERTADAPTION///////
  
  if (summe <= hisToUnknownThreshold) 
	  local_team_in_attack = HIS_TEAM; 
  else if (summe >  hisToUnknownThreshold && summe <= unknownToMyThreshold ) 
	  local_team_in_attack = UNKNOWN_TEAM;
  else  
	  local_team_in_attack = MY_TEAM;
  
  //be "selfish"
  if (WSinfo::is_ball_kickable())
  {
    local_teammate_last_at_ball_number = WSinfo::me->number;
    local_teammate_last_at_ball_time = WSinfo::ws->time;
  }
  else
  {
    pset_tmp = WSinfo::valid_teammates;
    pset_tmp.keep_players_in_circle(WSinfo::ball->pos, 2.0*ServerOptions::kickable_area);
    pset_tmp.keep_and_sort_closest_players_to_point(3,WSinfo::ball->pos);
    for (int i=0; i<pset_tmp.num; i++)
    {
      if (WSinfo::is_ball_kickable_for(pset_tmp[i]))
      {
        local_teammate_last_at_ball_number = pset_tmp[i]->number;
        local_teammate_last_at_ball_time = WSinfo::ws->time;
        break;
      }
    }
  }

  pset_tmp = WSinfo::valid_opponents;
  pset_tmp.keep_players_in_circle(WSinfo::ball->pos, 1.5*ServerOptions::kickable_area);
  pset_tmp.keep_and_sort_closest_players_to_point(3,WSinfo::ball->pos);
  for (int i=0; i<pset_tmp.num; i++)
  {
    if (WSinfo::is_ball_kickable_for(pset_tmp[i])) 
    {
      local_opponent_last_at_ball_number = pset_tmp[i]->number;
      local_opponent_last_at_ball_time = WSinfo::ws->time;
      break;
    }
  }
    
  if(local_teammate_last_at_ball_time >= local_opponent_last_at_ball_time)
    local_team_last_at_ball = 0;
  else
    local_team_last_at_ball = 1;

  if (WSinfo::is_ball_pos_valid()) 
  {
    saved_team_in_attack = local_team_in_attack;
    teammate_last_at_ball_number = local_teammate_last_at_ball_number;
    teammate_last_at_ball_time = local_teammate_last_at_ball_time;
    opponent_last_at_ball_number = local_opponent_last_at_ball_number;
    opponent_last_at_ball_time = local_opponent_last_at_ball_time;
    saved_team_last_at_ball = local_team_last_at_ball;
  }  
  else
  {
    //We need a fallback solution, even if ballpos is invalid.
    saved_team_in_attack = HIS_TEAM; 
    LOG_POL(1,<<"WSmemory: WARNUNG: Summe nicht korrekt berechenbar, weil ballPos invalid."); 
  }
  //debug output
  char teamName[10];
  if (saved_team_in_attack == HIS_TEAM) strcpy(teamName, "HIS_TEAM");
  else 
    if (saved_team_in_attack==MY_TEAM) strcpy(teamName, "MY_TEAM");
    else strcpy(teamName, "UNKNOWN");
  LOG_POL(1,<<_2D<<C2D( (((float)summe/3.0)*FIELD_BORDER_X+0.5),
                        FIELD_BORDER_Y+1.0, 0.5,"aaffaa"));
  char debugString[20];
  sprintf(debugString,"%.2f->%s",summe,teamName);
  LOG_POL(1,<<_2D<<STRING2D((((float)summe/3.0)*FIELD_BORDER_X+0.5),
                        FIELD_BORDER_Y+1.0,debugString,"aaffaa"));
  LOG_POL(2,<<"WSmemory: Summe = "<<summe<<" -> team_in_attack="<<teamName<<std::flush);
}

void WSmemory::update_offside_line_history() {
  if(last_update_at == WSinfo::ws->time) // ridi: do update only once per cycle
    return;
  his_offside_line_lag2 = his_offside_line_lag1;
  his_offside_line_lag1 = his_offside_line_lag0;
  his_offside_line_lag0 = WSinfo::his_team_pos_of_offside_line();
}

double WSmemory::get_his_offsideline_movement() {
#define TOLERANCE 0.4

  if(his_offside_line_lag0 > his_offside_line_lag1 + TOLERANCE)
    return 1.; // offsideline moves forward
  if(his_offside_line_lag0 < his_offside_line_lag1 - TOLERANCE)
    return -1.; // offsideline moves back;
  // no movement between now and previous, check lag 2
  if(his_offside_line_lag0 > his_offside_line_lag2 + TOLERANCE)
    return 1.; // offsideline moves forward
  if(his_offside_line_lag0 < his_offside_line_lag2 - TOLERANCE)
    return -1.; // offsideline moves backward
  return 0;
}

void WSmemory::update_his_goalie_classification()
{
  if ( last_update_at == WSinfo::ws->time )
    return;
  if ( WSinfo::ws->time == 0 )
    return;
  if ( WSinfo::his_goalie == NULL )
    return;
  if ( WSinfo::his_goalie->age > 0 )
    return;
  cvTimesHisGoalieSeen ++ ;
  if (    RIGHT_PENALTY_AREA.inside( WSinfo::his_goalie->pos) == false  
       && WSinfo::his_goalie->pos.getX() < FIELD_BORDER_X + 1.0
       && WSinfo::his_goalie->pos.distance(WSinfo::ball->pos) < 7.0 )
    cvTimesHisGoalieOutsidePenaltyArea ++ ;
  LOG_POL(0,<<"WSmemory: cvTimesHisGoalieOutsidePenaltyArea="
    <<cvTimesHisGoalieOutsidePenaltyArea<<" cvTimesHisGoalieSeen="
    <<cvTimesHisGoalieSeen
    <<" (last_update_at="<<last_update_at<<", time="<<WSinfo::ws->time
    <<", his_goalie="<<WSinfo::his_goalie<<")");
}

int  WSmemory::get_his_goalie_classification()
{
  double offensiveShare =   (double)cvTimesHisGoalieOutsidePenaltyArea
                         / (double)cvTimesHisGoalieSeen;
  if ( cvTimesHisGoalieSeen < 20 )
    return HIS_GOALIE_DEFENSIVE;
  if (    offensiveShare > 0.05
       || (   cvTimesHisGoalieSeen > 200
           && cvTimesHisGoalieOutsidePenaltyArea > 10 ) )
  {
    return HIS_GOALIE_OFFENSIVE;
  }
  return HIS_GOALIE_DEFENSIVE;
}

void WSmemory::update() {
  if(WSinfo::ws->time_of_last_update == WSinfo::ws->time) {
    add_view_info(WSinfo::ws->time,WSinfo::ws->view_angle,
		  WSinfo::ws->view_quality,WSinfo::me->neck_ang,WSinfo::me->pos);
    add_attentionto_info(WSinfo::ws->time,WSinfo::ws->my_attentionto);
  }
  update_our_selfpass_memory();
  if(WSinfo::is_ball_kickable())
    ball_was_kickable4me_at = WSinfo::ws->time;
  update_fastest_to_ball();
  update_offside_line_history();
  update_his_goalie_classification();
  update_our_selfpass_assessment();
  last_update_at = WSinfo::ws->time;
  if (    WSinfo::ws->play_mode == PM_my_KickOff 
       || WSinfo::ws->play_mode == PM_my_BeforeKickOff
       || WSinfo::ws->play_mode == PM_his_BeforeKickOff
       || WSinfo::ws->play_mode == PM_his_KickOff )
    last_kick_off_at = WSinfo::ws->time;
  if ( WSinfo::ws->play_mode == PM_his_GoalKick )
    cvLastHisGoalKick = WSinfo::ws->time;
}

void WSmemory::add_view_info(long time,int vang,int vqty,ANGLE vdir,Vector ppos) {
  if(vqty == Cmd_View::VIEW_QUALITY_LOW) return;
  for(int i=MAX_VIEW_INFO-2;i>=0;i--) {
    view_info[i+1]=view_info[i];
  }
  view_info[0].time=time;
  view_info[0].view_width=Tools::get_view_angle_width(vang);
  view_info[0].view_dir=vdir;
  view_info[0].ppos=ppos;
}

void WSmemory::add_attentionto_info(long time, int plNr)
{
  for ( int i=MAX_ATTENTIONTO_INFO-2; i>=0 ;i-- ) 
  {
    attentionto_info[i+1] = attentionto_info[i];
  }
  attentionto_info[0].time = time;
  attentionto_info[0].playerNumber = plNr;
}

long WSmemory::last_attentionto_to(int plNr)
{
  for (int i=0; i<MAX_ATTENTIONTO_INFO; i++)
  {
    if (attentionto_info[i].time < 0) continue;
    if (attentionto_info[i].playerNumber == plNr)
      return WSinfo::ws->time - attentionto_info[i].time;
  }
  return (WSinfo::ws->time +1);
}

long WSmemory::last_seen_in_dir(ANGLE dir) {
  if(WSinfo::ws->ms_time_of_see<0)
    return WSinfo::ws->time - WSinfo::ws->time_of_last_update;
  for(int i=0;i<MAX_VIEW_INFO;i++) {
    if(view_info[i].time < 0) continue;
    double dir_diff = fabs((view_info[i].view_dir-dir).get_value_mPI_pPI());
    if(dir_diff < .5*view_info[i].view_width.get_value()) {
      return WSinfo::ws->time - view_info[i].time;
    }
  }
  return (WSinfo::ws->time +1); // ridi: return time+1 instead of -1
}
    
long WSmemory::last_seen_to_point(Vector target) {
  if(WSinfo::ws->ms_time_of_see<0)
    return WSinfo::ws->time - WSinfo::ws->time_of_last_update;
  for(int i=0;i<MAX_VIEW_INFO;i++) {
    if(view_info[i].time < 0) continue;
    if((target-view_info[i].ppos).sqr_norm()>SQUARE(MAX_VISIBLE_DISTANCE)) continue;
    ANGLE tmp_dir = (target-view_info[i].ppos).ARG();
    double dir_diff = fabs((view_info[i].view_dir-tmp_dir).get_value_mPI_pPI());
    if(dir_diff < .5*view_info[i].view_width.get_value()) {
      return WSinfo::ws->time - view_info[i].time;
    }
  }
  return (WSinfo::ws->time +1); // ridi: return time+1 instead of -1
}

bool 
WSmemory::get_view_info_before_n_steps(int n,
                                       ANGLE  & viewWidth,
                                       ANGLE  & viewDir,
                                       Vector & playerPos,
                                       long   & globalTime)
{
  if (n >= MAX_VIEW_INFO) 
  return false;
  viewWidth = view_info[n].view_width;
  viewDir   = view_info[n].view_dir;
  playerPos = view_info[n].ppos;
  globalTime= view_info[n].time;
  return true;
}


/* I won't implement this without having PolicyTools independent of mdpInfo! */
#if 0
/* code taken from MDPmemory::update(), should be checked for correctness! */
void WSmemory::update_attack_info() {
  Vector ball, player;
  if(WSinfo::is_ball_pos_valid() {
    int summe = 0;
#endif

void WSmemory::update_our_selfpass_memory()
{
  if (   Blackboard::main_intention.get_type() == SELFPASS
      && (   (   teammate_last_at_ball_time == WSinfo::ws->time - 1
              && teammate_last_at_ball_number == WSinfo::me->number )
          || WSinfo::is_ball_pos_valid() ) 
     )
  {
    LOG_POL(0,"WSmemory: SELFPASSMEMORY: last time step i played a self pass");
    cvOurLastSelfPass = WSinfo::ws->time - 1;
    if (   WSinfo::is_ball_kickable() == false
        && (   (   teammate_last_at_ball_time == WSinfo::ws->time - 1
              && teammate_last_at_ball_number == WSinfo::me->number ) )
        )
    {
      cvOurLastSelfPassWithBallLeavingKickRange = WSinfo::ws->time - 1;
      cvOurLastSelfPassRiskLevel = Selfpass2::getRiskLevel();
    }
  }
  LOG_POL(0,"WSmemory: SELFPASSMEMORY: tmmL@B="<<teammate_last_at_ball_time
    <<"tmmL@Bnr="<<teammate_last_at_ball_number<<" isBK="<<WSinfo::is_ball_kickable()
    <<" oppL@B="<<opponent_last_at_ball_time<<" oppL@Bnr="<<opponent_last_at_ball_number);
  LOG_POL(0,"WSmemory: SELFPASSMEMORY: last selfpass: "<<cvOurLastSelfPass);
  LOG_POL(0,"WSmemory: SELFPASSMEMORY: last selfpass leaving KR: "<<cvOurLastSelfPassWithBallLeavingKickRange);
}

void WSmemory::update_our_selfpass_assessment()
{
  if ( cvOurLastSelfPassWithBallLeavingKickRange < 0 )
    return; //wurde schon ausgewertet
  if (    teammate_last_at_ball_time <= cvOurLastSelfPassWithBallLeavingKickRange
       && opponent_last_at_ball_time <= cvOurLastSelfPassWithBallLeavingKickRange )
    return; //selfpass laeuft noch
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_and_sort_closest_players_to_point(1, WSinfo::ball->pos);
  bool anOpponentHasExtremelyHighTackleProbability = false;
  if (    opps.num > 0 
       && opps[0]->pos.distance(WSinfo::ball->pos) < 2.0
       && opps[0]->age==0 && opps[0]->age_ang==0 )
  {
    double  tackleSuccessProbability
      = Tools::get_tackle_success_probability( opps[0]->pos,
                                               WSinfo::ball->pos,
                                               opps[0]->ang.get_value());
    if (tackleSuccessProbability > 0.8)
    {
      anOpponentHasExtremelyHighTackleProbability = true;
      LOG_POL(0,"WSinfo: SELFPASSMEMORY: WARNING, oponent "<<opps[0]->number
        <<" has HIGH tackle probability: "<<tackleSuccessProbability);
    }
    else
    {
      LOG_POL(0,"WSinfo: SELFPASSMEMORY: NOTE, opponent "<<opps[0]->number
        <<" has tackle probability of "<<tackleSuccessProbability);
    }    
  }
  if (    opponent_last_at_ball_time == WSinfo::ws->time
       || anOpponentHasExtremelyHighTackleProbability )
  {
    //selbstpass ist schief gegangen
    cvSelfpassAssessment[cvOurLastSelfPassRiskLevel][SELFPASS_FAILURE] ++ ;
    cvOurLastSelfPassWithBallLeavingKickRange = -1;
  }
  else
  if ( teammate_last_at_ball_time == WSinfo::ws->time )
  {
    if ( WSinfo::is_ball_kickable() == true )
    {
      //selbstpass hat geklappt
      cvSelfpassAssessment[cvOurLastSelfPassRiskLevel][SELFPASS_SUCCESS] ++ ;
      cvOurLastSelfPassWithBallLeavingKickRange = -1;
    }
    else
    {
      //selbstpass ist bei einem mitspieler angekommen: Egal!      
      cvOurLastSelfPassWithBallLeavingKickRange = -1;
      cvSelfpassAssessment[cvOurLastSelfPassRiskLevel][SELFPASS_DONTCARE] ++ ;
    }
  }  
  LOG_POL(0,<<"WSmemory: SELFPASSASSESSMENT [tmmL@B="<<teammate_last_at_ball_time
    <<",oppL@B="<<opponent_last_at_ball_time<<"]: ");
  int maxRiskLevel = 0;
  for (int i=0; i<MAX_SELFPASS_RISK_LEVELS; i++)
    if (   cvSelfpassAssessment[i][SELFPASS_DONTCARE] > 0 
        || cvSelfpassAssessment[i][SELFPASS_FAILURE] > 0 
        || cvSelfpassAssessment[i][SELFPASS_SUCCESS] > 0)
      maxRiskLevel = i;
  for (int i=0; i<=maxRiskLevel; i++)
  {
    LOG_POL(0,"          risk level "<<i<<": [F]"<<cvSelfpassAssessment[i][SELFPASS_FAILURE]
                                        <<"  [DC]"<<cvSelfpassAssessment[i][SELFPASS_DONTCARE]
                                        <<"  [S]"<<cvSelfpassAssessment[i][SELFPASS_SUCCESS]);
  }
  return;
}

bool
WSmemory::getSelfpassMemoryForRiskLevel( int riskLevel,
                                         int & numberOfSuccessfulSelfpasses,
                                         int & numberOfUndecidedSelfpasses,
                                         int & numberOfFaultySelfpasses
                                       )
{
  if (riskLevel < 0 || riskLevel >= MAX_SELFPASS_RISK_LEVELS)
    return false;
  numberOfSuccessfulSelfpasses = cvSelfpassAssessment[riskLevel][SELFPASS_SUCCESS];
  numberOfUndecidedSelfpasses = cvSelfpassAssessment[riskLevel][SELFPASS_DONTCARE];
  numberOfFaultySelfpasses = cvSelfpassAssessment[riskLevel][SELFPASS_FAILURE];
  return true;
}

void
WSmemory::incorporateCurrentCmd( const Cmd& cmd )
{
  if ( cmd.cmd_body.get_type() == Cmd_Body::TYPE_TACKLE )
  {
    double dir, foul;
    cmd.cmd_body.get_tackle(dir, foul);
    if ((int)foul == 1)
    {
      cvLastTimeFouled = WSinfo::ws->time;
      cvLastFouledOpponentNumber = opponent_last_at_ball_number;
      LOG_POL(0, << "WSmemory: I remember that I do play foul now at t="<<cvLastTimeFouled
        << ", fouling opp #"<<cvLastFouledOpponentNumber);
    }
  }
}
