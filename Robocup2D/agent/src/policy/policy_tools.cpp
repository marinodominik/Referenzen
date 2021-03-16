#include "policy_tools.h"
#include "mdp_info.h"
#include "tools.h"
#include "log_macros.h"
#include "options.h"
#include "valueparser.h"
#include "intercept.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define CONSIDER_AGE_MAXINCREASE .5
#define BASELEVEL 0
//#define SQUARE(x)((x)*(x))
#define IGNORE_OPS_SAFETY 10.  // the smaller,the more ops are disregarded. >5: seems to be reasonable
#define SELFPASS_IGNORE_OPS_SAFETY 15.  // the smaller,the more ops are disregarded. >=10:
#define INFINITE_STEPS 2000
#if 0
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#endif

/*****************************************************************************/

void MatchPositions::match(const Vector* players, int num_players,
			   const Vector* targets, int num_targets) {
  if (num_targets> num_targets_MAX || num_targets < 0) {
    num_targets= num_targets_MAX;
    cerr << "\nMatchPositions::match to many targets";
  }

  if (num_players> num_players_MAX || num_players < 0) {
    num_players= num_players_MAX;
    cerr << "\nMatchPositions::match to many players";
  }
  
#if 0
  cerr << "\n->>>>" 
       << "\n num_players= " << num_players;
  for (int i=0; i< num_players; i++)
    cerr << " " << players[i];

  cerr << "\n num_targets= " << num_targets;
  for (int i=0; i< num_targets; i++)
    cerr << " " <<  targets[i];
#endif

  //compute nearest players for all targets!
  for (int t=0; t<num_targets; t++) 
    for (int p=0; p<num_players; p++) {
      double d= targets[t].distance( players[p] );
      if (d < nearest_from_target[t][0].dist) {
	nearest_from_target[t][2]= nearest_from_target[t][1];
	nearest_from_target[t][1]= nearest_from_target[t][0];
	nearest_from_target[t][0]= Dist_2_Player(p,d);
      }
      else if (d < nearest_from_target[t][1].dist) {
	nearest_from_target[t][2]= nearest_from_target[t][1];
	nearest_from_target[t][1]= Dist_2_Player(p,d);
      }
      else if (d < nearest_from_target[t][2].dist) 
	nearest_from_target[t][2]= Dist_2_Player(p,d);
    }

#if 0
  //output of nearest neighbours
  for (int t=0; t<num_targets; t++) {
    cerr << "\ntarget= " << t;
    for (int n=0; n<3; n++) {
      cerr << " (" << nearest_from_target[t][n].player
	   << ","  << nearest_from_target[t][n].dist << ")";
    }
  }
#endif

  int  player_in_sol[num_players];
  //set all players to be contained in solutioin number -1 (i.e. be free)
  for (int p= 0; p<num_players; p++) player_in_sol[p]= -1;

  
  int num_solutions= 1;
  for (int t=0; t<num_targets; t++) num_solutions*= 3;
  // num_solutions =  3 ^ num_targets


  int  num_players_of_best_sol= 0;
  double cost_of_best_sol= -1.0;
  int idx_of_best_sol= -1;

  for (int idx_of_sol=0; idx_of_sol <num_solutions; idx_of_sol++) {
    int tmp_sol= idx_of_sol;
    int num_players_of_sol = 0;
    double cost_of_sol= 0.0;

    //cerr << "\nsol= " << idx_of_sol;
    for (int t= 0; t< num_targets; t++) {
      int idx= tmp_sol%3;
      tmp_sol /= 3;
      int p= nearest_from_target[t][idx].player;      
      //cerr << " nearest_from_target[t][idx].player= " << p;      
      // if player_in_sol[p]==idx_of_sol  then this player was already taken for this solution
      if ( p < 0 || player_in_sol[p]==idx_of_sol ) 
	continue;
      else {
	num_players_of_sol += 1;
	player_in_sol[p]= idx_of_sol;
	cost_of_sol += nearest_from_target[t][idx].dist;
      }
    }
    //cerr << "\ncost= " << cost_of_sol;
    if (num_players_of_sol <= 0) continue;
    if ((cost_of_sol < cost_of_best_sol && num_players_of_sol >= num_players_of_best_sol)
	|| num_players_of_sol > num_players_of_best_sol
	|| cost_of_best_sol < 0.0 ) {
      num_players_of_best_sol= num_players_of_sol;
      cost_of_best_sol= cost_of_sol;
      idx_of_best_sol= idx_of_sol;
    }
  }
  
#if 0
  cerr << "\nNum  solution       = " << num_solutions;
  cerr << "\nBest solution       = " << idx_of_best_sol;
  cerr << "\nBest solution cost  = " << cost_of_best_sol;
  cerr << "\nplayers of best sol = " << num_players_of_best_sol;
#endif
  //init array for best pos match
  for (int p= 0; p<num_players; p++)
    best_sol[p]= -1;

  if (idx_of_best_sol < 0)
    return;

  int tmp_sol= idx_of_best_sol;

  for (int t= 0; t< num_targets; t++) {
    int idx= tmp_sol%3;
    int p= nearest_from_target[t][idx].player;
    if ( best_sol[p] == -1)
      best_sol[p]= t;
    tmp_sol /= 3;
  }

}
/*****************************************************************************/

void PlayerMatch::init(int index, 
		       const Vector me,
		       const Vector* targets, int num_targets){
  mapped_to = -1;
  my_index = index;
  shortest_index = -1;
  distances = new double[num_targets];
  for (int tg =0 ; tg < num_targets ; tg++){
    distances[tg] = me.distance(targets[tg]);
  }
}

MatchPositionsAlternative::MatchPositionsAlternative(int num_players, int num_targets){
  n_players = num_players;
  n_targets = num_targets;
  map = new int[num_players];
  player = 0;
  temp_max = 0;
}

void MatchPositionsAlternative::match(const Vector* players, int num_players,
				      const Vector* targets, int num_targets){
  int pl =0, pos=0;
  player = new PlayerMatch[num_players];
  for (int i=0; i<num_players ; i++){
    player[i].init(i, players[i], targets, num_targets);
  }
  for (int i=0 ; i<num_players ; i++){
    for (pl=0 ; pl<num_players ; pl++){
      if (player[pl].mapped_to < 0)
	player[pl].shortest_index = min_element(player[pl].distances,num_targets);
    } 
    temp_max = max_element(player, num_players);
    player[temp_max].mapped_to=player[temp_max].shortest_index;
    //    cout << "Selected : Player " << temp_max << " at position "<< player[temp_max].mapped_to <<endl;
    for (pos = 0; pos<num_targets ; pos++){
      player[temp_max].distances[pos]=0;
    }
    for (pl=0 ; pl<num_players ; pl++){
      if (player[pl].mapped_to < 0){
	player[pl].distances[player[temp_max].mapped_to]=10000;
      }
    }
  }
  for (pl = 0 ; pl < num_players ; pl++){
    map[pl] = player[pl].mapped_to;
  }
}

int MatchPositionsAlternative::min_element(double *array, int array_length){
  int min = 0;
  for (int i=1 ; i < array_length ; i++){
    if (array[min] > array[i]){
      min = i;
    }
  }
  return min;
}

int MatchPositionsAlternative::max_element(PlayerMatch *array, int array_length){
  int max = 0;
  while ((array[max].mapped_to >=0 ) && (max <array_length-1)){
    max++;
  }
  for (int i=max+1 ; i < array_length ; i++){
    if (array[max].distances[array[max].shortest_index] < array[i].distances[array[i].shortest_index]){
      max = i;
    }
  }
  return max;
}


/*****************************************************************************/

void MatchDistribution::exchange(int i,int j, val2ord * v) {
  val2ord tmp= v[i];
  v[i]= v[j];
  v[j]= tmp;
}


void MatchDistribution::qsort(int size,val2ord * v) {
  if (size < 8) {
    for (int i=0; i< size; i++)
      for (int j=i; j<size;j++) {
        if ( v[i].value > v[j].value ) exchange(i,j,v);
      }
    return;
  }
  
  int beg= 0;
  int end= size-1;
  float app_median=  0.5*(v[beg].value+v[end].value);
  while (true) {
    while (v[beg].value< app_median && beg < size) { beg++;}
    while (v[end].value>= app_median && end >= 0) { end--;}
    if (beg < end)
      exchange(beg,end,v);
    else {
      if (beg==0) beg=1;
      if (beg==size) beg= size-1;
      qsort(size-beg,v+beg);
      qsort(beg,v);
      return;
    }
  }
}

void MatchDistribution :: convert(float * values, int num_players, int num_positions){
  v = new val2ord[num_players*num_positions];
  int tplayers=0;
  int tpos=0;

  for(int i=0;i<num_players*num_positions;i++){
    v[i].value = values[i];
    v[i].player = tplayers;
    v[i].position = tpos;
    tpos++;
    if(tpos == num_positions){
      tplayers++;
      tpos = 0;
    }
  }  

}


void MatchDistribution::match(float * values, int num_players, int num_positions){
  map = new int[num_players];
  int *tmap = new int[num_positions];
  int tpos[num_positions];
  int tplayer[num_players];

  for(int i=0;i<num_players;i++){
    map[i] = -1;
  }
  for(int i=0;i<num_positions;i++){
    tmap[i] = -1;
  }

  for(int i=0;i<num_players;i++){
    tplayer[i] = num_positions;
  }

  for(int i=0;i<num_positions;i++){
    tpos[i] = num_players;
  }

  convert(values, num_players, num_positions);
  qsort(num_players*num_positions, v);
  
  for(int i=num_players*num_positions-1;i>=0;i--){
    //    if (mdpInfo::mdp->me->number == 2) cout << "\nPlayer: " << v[i].player << " Pos: " << v[i].position << " val: " << v[i].value;
    tplayer[v[i].player]--;
    tpos[v[i].position]--;
    if(tplayer[v[i].player]==0 || tpos[v[i].position]==0){
      //      if(mdpInfo::mdp->me->number == 2) cout << "\nPlayer: " << v[i].player << " goes " << v[i].position;
      if(map[v[i].player]==-1 && tmap[v[i].position]==-1){
        map[v[i].player] = v[i].position;
        tmap[v[i].position] = v[i].player;
      }
    }
  }
  for(int i=0;i<num_players;i++){
    //if(mdpInfo::mdp->me->number == 2) cout << "\nPlayer: " << i+2 << " Pos: " << map[i];
  }
}

/*****************************************************************************/


double Policy_Tools::goalshot_goalie_bonusstep;
double Policy_Tools::goalshot_consider_age;
double Policy_Tools::goalshot_worstcase_consider_age;
double Policy_Tools::extremeshot_goalie_maxspeed_percentage;
double Policy_Tools::extremeshot_goalie_playersize_percentage;
double Policy_Tools::goalshot_corner_tolerance;
bool Policy_Tools::go2ball_list_analytical=false; //default: use neural value function
Go2Ball_Steps Policy_Tools::go2ball_list[22];
bool Policy_Tools::turnneck2ballholder;
bool Policy_Tools::use_go2ball_classic;
int Policy_Tools::pass_mytime2react;
int Policy_Tools::selfpass_optime2react;
int Policy_Tools::selfpass_my_minadvantage;
int Policy_Tools::pass_optime2react;
int Policy_Tools::pass_my_minadvantage;

int Policy_Tools::last_go2ball_steps_update = -1;
int Policy_Tools::last_intercept_pos_update = -1;
Vector Policy_Tools::saved_intercept_pos = Vector(0.0, 0.0);

float Policy_Tools::defense_area = 16;
bool Policy_Tools::use_clever_moves=false;
bool Policy_Tools::goaliestandards=false;
Vector Policy_Tools::ballpos;

void Policy_Tools::init_params(){

  // default settings
  goalshot_goalie_bonusstep = 2;  // goalie is 1 step closer to ball
  extremeshot_goalie_maxspeed_percentage =.7; 
  extremeshot_goalie_playersize_percentage =.0; 
  goalshot_corner_tolerance =1.7;
  goalshot_consider_age =0;
  goalshot_worstcase_consider_age =0.8;
  turnneck2ballholder= true;
  use_go2ball_classic = true;
  defense_area = 16;
  pass_mytime2react=3;
  selfpass_optime2react=1;
  selfpass_my_minadvantage=2;
  pass_optime2react=1; // was 0
  pass_my_minadvantage=1; // was 1
  ValueParser vp( CommandLineOptions::policy_conf, "Policy_Tools" );
  //  vp.set_verbose(true);
  vp.get("goalshot_goalie_bonusstep",goalshot_goalie_bonusstep);
  vp.get("goalshot_consider_age",goalshot_consider_age);
  vp.get("goalshot_worstcase_consider_age",goalshot_worstcase_consider_age);
  vp.get("extremeshot_goalie_maxspeed_percentage",extremeshot_goalie_maxspeed_percentage);
  vp.get("extremeshot_goalie_playersize_percentage",extremeshot_goalie_playersize_percentage);
  vp.get("goalshot_corner_tolerance",goalshot_corner_tolerance);
  vp.get("go2ball_list_analytical",go2ball_list_analytical);
  vp.get("turnneck2ballholder",turnneck2ballholder);
  vp.get("use_go2ball_classic",use_go2ball_classic);
  vp.get("pass_mytime2react",pass_mytime2react);
  vp.get("selfpass_optime2react",selfpass_optime2react);
  vp.get("selfpass_my_minadvantage",selfpass_my_minadvantage);
  vp.get("pass_optime2react",pass_optime2react);
  vp.get("pass_my_minadvantage",pass_my_minadvantage);
}

void  Policy_Tools::go2ball_steps_update(){

  if (last_go2ball_steps_update == WSinfo::ws->time) {
    return;
  }

  last_go2ball_steps_update = WSinfo::ws->time;
  
  Vector ball_vel = WSinfo::ball->vel;

  //D Neuro_State state;
  //D  Neuro_Intercept_Ball *intercept = Move_Factory::get_Eval_Neuro_Intercept();
  int n=0;
  float steps;

  //D state.ball_pos = WSinfo::ball->pos;
  //D state.ball_vel = WSinfo::ball->vel;

  PlayerSet pset_tmp = WSinfo::valid_opponents;
  PPlayer p_tmp = pset_tmp.closest_player_to_point(WSinfo::ball->pos);
  
  if (p_tmp != NULL && WSinfo::is_ball_kickable_for(p_tmp)) {
    //Daniel
    //if opponent can kick ball, don't use ball velocity, because we don't know if he is going to pass the
    //ball or keep it in his kick_range
    ball_vel = Vector(0.0, 0.0);
  }
  
  for(int i=1; i<=11; i++){
    p_tmp = WSinfo::get_teammate_by_number(i);
    if (p_tmp != NULL) {
      //D state.my_pos = p_tmp->pos;
      //D state.my_vel = p_tmp->vel;
      //D state.my_angle = (WSinfo::ball->pos-WSinfo::me->pos).arg();     
      
      int tmp_steps;
      Vector my_intercept_pos; // not used here
      intercept_min_time_and_pos_hetero(tmp_steps, my_intercept_pos, WSinfo::ball->pos, 
					ball_vel, p_tmp->pos, i, true, -1.0, -1000.0); 
      steps = tmp_steps;

      go2ball_list[n].side = Go2Ball_Steps::MY_TEAM;
      go2ball_list[n].number = i;
      go2ball_list[n].steps = steps;
      
      n++;
      
    }
    p_tmp = WSinfo::get_opponent_by_number(i);
    if (p_tmp != NULL) {
      //D state.my_pos = p_tmp->pos;
      //D state.my_vel = p_tmp->vel;
      //D state.my_angle = (state.ball_pos-state.my_pos).arg();     
     

      int tmp_steps;
      Vector my_intercept_pos; // not used here
      intercept_min_time_and_pos_hetero(tmp_steps, my_intercept_pos, WSinfo::ball->pos, 
					ball_vel, p_tmp->pos, i, false, -1.0, -1000.0); 
      steps = tmp_steps;


      go2ball_list[n].side = Go2Ball_Steps::THEIR_TEAM;
      go2ball_list[n].number = i;
      go2ball_list[n].steps = steps;
      n++;
      //DBLOG_POLICY << "Op: "<< i << " steps: " << steps;

    }
  }
  
  // sort go2ball steps list
  for(int i=0;i<n-1;i++){
    for(int j=i+1;j<n;j++){
      int min_idx = i;
      if(go2ball_list[j].steps < go2ball_list[i].steps){
	min_idx = j;
      }
      Go2Ball_Steps save = go2ball_list[i];
      go2ball_list[i] = go2ball_list[min_idx];
      go2ball_list[min_idx] = save;
    }
  }  
 

  for(int i=n;i<22;i++){
    go2ball_list[i].side = Go2Ball_Steps::NO_TEAM;
  }

}


Go2Ball_Steps* Policy_Tools::go2ball_steps_list(){
  return go2ball_list;
}


Vector Policy_Tools::next_intercept_pos(double speed_factor ){
  if (last_intercept_pos_update == WSinfo::ws->time) {
    return saved_intercept_pos;
  }
  last_intercept_pos_update = WSinfo::ws->time;

  PlayerSet pset_tmp = WSinfo::valid_teammates;
  pset_tmp += WSinfo::valid_opponents;

  for (int i = 0; i < pset_tmp.num; i++) {
    if (WSinfo::is_ball_kickable_for(pset_tmp[i]))
      return pset_tmp[i]->pos;
  }

  Vector ball_vel = WSinfo::ball->vel;
  Vector ball_pos = WSinfo::ball->pos;
  Vector intercept_pos;
  bool myteam;
  int number;

  Policy_Tools::earliest_intercept_pos(ball_pos, ball_vel.norm(), ball_vel.arg(), 
				       intercept_pos, myteam, number, speed_factor);
  saved_intercept_pos = intercept_pos;
  return intercept_pos;
}


bool Policy_Tools::earliest_intercept_pos(const Vector ballpos,const double speed, const double dir,
					  Vector &intercept_pos,bool &myteam, int &number, 
					  double speed_factor) {
  int advantage; 

  return earliest_intercept_pos(ballpos, speed,dir,intercept_pos,myteam,number,advantage,speed_factor);
}

bool Policy_Tools::earliest_intercept_pos(const Vector ballpos,const double speed, const double dir,
					  Vector &intercept_pos,bool &myteam, int &number, 
					  int &advantage, double speed_factor) {
#define OUTOF_PITCH_SAFETY 7.0
  Vector velocity;
  velocity.init_polar(speed,dir);
  /*
    Vector add_pos = velocity; // this is a help vector
    add_pos.normalize(1.1); // put ball outside kickrange
  */

  int fastest_steps = 2000;
  Vector earliest_intercept = Vector(0);
  myteam = false; // be pessimistic

  int mytime2react =1;// = 0;
  int optime2react =1; // = 0;

  PlayerSet pset = WSinfo::valid_teammates;

  for (int idx = 0; idx < pset.num; idx++) {
    int steps;
    Vector player_intercept_pos;
    if(pset[idx]->number == WSinfo::ws->my_goalie_number) // do not consider goalie as interceptor
      continue;
    steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
			       speed,dir,  pset[idx],
			       mytime2react,fastest_steps); 
    if(steps <0) // this means that player doesnt get ball at all
      continue; 

    if(steps < fastest_steps) { // the fastest yet
      fastest_steps = steps;
      //myteam = false; ridi: this was wrong!
      number = pset[idx]->number;
      earliest_intercept = player_intercept_pos;
    }
  }


  int myteam_fastest = fastest_steps;

  fastest_steps = 2000; // reinit for opponent

  pset = WSinfo::valid_opponents;

  for (int idx = 0; idx < pset.num; idx++) {
    int steps;
    Vector player_intercept_pos;
    steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
			       speed,dir,  pset[idx],
			       optime2react,fastest_steps); 
    
    if(steps <0) // this means that player doesnt get ball at all
      continue; 

    if (steps < fastest_steps) { // the fastest opponent yet
      fastest_steps = steps;
      if(steps < myteam_fastest){ // its the best opponent and its better than my teams fastest
	earliest_intercept = player_intercept_pos;
      }
    }
  }

  advantage = fastest_steps - myteam_fastest;
  if (advantage >0)
    myteam = true;

  intercept_pos = earliest_intercept;
#if 1
  DBLOG_POL(1,<<"PT: Earliest intercept pos for dir "<<RAD2DEG(dir)<<" speed "<<speed<<" ballpos "
	    <<ballpos<<" ipos: "
	    <<intercept_pos<<" advantage "<<advantage<<" my fastest "<<number<<" op fastest "<<op_number);
#endif
  if (Tools::is_position_in_pitch(earliest_intercept, OUTOF_PITCH_SAFETY))
    return true;
  else
    return false;
}

void Policy_Tools::check_if_turnneck2ball(int teammate_steps2ball, 
					  int opponent_steps2ball){
  if(turnneck2ballholder == false)
    return;
  //  if(teammate_steps2ball>0 && opponent_steps2ball >0)
#if 0
  if(teammate_steps2ball==1)
    DBLOG_ERR(0,"Teammate has ball.: ");
  if(opponent_steps2ball==1)
    DBLOG_ERR(0,"OP. has ball.");
#endif
  if(teammate_steps2ball>1 && opponent_steps2ball >1)
    return; // nobody has the ball
  if(mdpInfo::is_ball_infeelrange())
    return; // I can feel it!
  Vector rel_ballpos = WSinfo::ball->pos - WSinfo::me->pos;
  if(rel_ballpos.norm() > 100.)  // ball is too far away
    return;
#if 0
  // ridi: do not check this, since I migth do a turn command in this cyle
  // just try to turn to that direction
  if(mdpInfo::could_see_in_direction(rel_ballpos.arg()) == false)
    return;
#endif
  //DBLOG_ERR(0,"Player has ball. Turn neck 2 ball, Dir: "<<RAD2DEG(rel_ballpos.arg()));
  //DBLOG_POL(4,"Player has ball. Turn neck 2 ball, Dir: "<<RAD2DEG(rel_ballpos.arg()));
  mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION,rel_ballpos.arg() );
}


bool Policy_Tools::get_best_kicknrush(const Vector ballpos, const int testdirs,
				      const double testdir[], double testspeed[],
				      double &speed, double &dir, Vector &resulting_pos){
  int advantage;
  int number;

  
  return  get_best_kicknrush(ballpos, testdirs, testdir, testspeed, speed,dir, resulting_pos,
			     advantage,number);
}

bool Policy_Tools::get_best_kicknrush(const Vector ballpos, const int testdirs,
				      const double testdir[], double testspeed[],
				      double &speed, double &dir, Vector &resulting_pos,
				      int &advantage, int &closest_teammate)
{

  Vector bestipos = Vector(-100,0);
  bool pass2myteam = false;
  double bestdir;
  double bestspeed;
  int bestadvantage = -100;
  int bestnumber = 0;
  float best_dist2playerdir = PI;

  if(testdirs >0)
  {
    bestdir = testdir[0]; // default
    bestspeed = testspeed[0];
  }
  else
  {
    bestdir = 0;
    bestspeed = 2.5;
  }

  
  for (int j=0; j<testdirs; j++)
  {
    Vector ipos;
    bool myteam;
    int number;
    int tmp_advantage;

    if( (testspeed[j]>2.5) || testspeed[j] <0)
    {
      testspeed[j] = 2.5;
    }

    bool inpitch=Policy_Tools::earliest_intercept_pos(ballpos,testspeed[j],testdir[j],
						      ipos, myteam, number, tmp_advantage);

#if 0
    DBLOG_POL(1,<<"PT: Testing Kicknrush "<<" dir "<<RAD2DEG(testdir[j])
	      <<" intercept pos "<<ipos<<" myteam intercepts "<<myteam<<" player number "
	      <<number<<" advantage "<<tmp_advantage<<" in pitch "<<inpitch);
#endif

    if (!inpitch)
      continue;
    if (RIGHT_PENALTY_AREA.inside(ipos) == true && myteam == false)
      continue;
    if (number == WSinfo::me->number)
      myteam = false; // just set myteam to false. Risky selfpasses are handled elsewhere
    //      continue;
    if (number == WSinfo::ws->my_goalie_number)
      continue;
 
    DBLOG_POL(0,<<"Pass2myteam "<<pass2myteam<<" myteam "<<myteam);
    if (pass2myteam == false)
    { // I haven't found a possibility to pass to my team member yet
      if (
              ((ipos.getX() > bestipos.getX()) || (myteam == true))
           && isKicknRushInterceptionPointAcceptable( ipos, tmp_advantage )
         )
      {
      	bestipos = ipos;
      	bestdir = testdir[j];
      	bestspeed = testspeed[j];
      	bestadvantage = tmp_advantage;
      	bestnumber = number;
      	if (myteam == true && number != WSinfo::me->number)
        {
    	  //DBLOG_ERR(BASELEVEL,"kicknrush: A team player gets the ball");
    	    PPlayer p= WSinfo::get_teammate_by_number(number);
    	    if(p)
          {
      	    ANGLE playerdir = (p->pos - WSinfo::ball->pos).ARG();
    	      best_dist2playerdir = playerdir.diff(ANGLE(testdir[j]));
    	    }
    	    pass2myteam = true;
    	  }
      }
    }
    else
    {  // the best position already goes to my player, so try to improve if possible
      float dist2playerdir;
      PPlayer p= WSinfo::get_teammate_by_number(number);
      if (p)
      {
        ANGLE playerdir = (p->pos - WSinfo::ball->pos).ARG();
        dist2playerdir = playerdir.diff(ANGLE(testdir[j]));
      }
      else
        dist2playerdir = 2*PI; // max

      DBLOG_POL(0,<<"PT: Kicknrush: pass in dir "<<RAD2DEG(testdir[j])<<" interceptor "<<number
  		<<" dist2playerdir "<<RAD2DEG(dist2playerdir)
	  	<<" best dist2player dir "<<RAD2DEG(best_dist2playerdir));
      
      // ridi: optimization criterion: as close to the target player as possible
      if (    (dist2playerdir < best_dist2playerdir) && (myteam == true)
           && isKicknRushInterceptionPointAcceptable( ipos, tmp_advantage )
         )
      {
      	best_dist2playerdir = dist2playerdir;
      	bestipos = ipos;
      	bestdir = testdir[j];
      	bestspeed = testspeed[j];
      	bestadvantage = tmp_advantage;
      	bestnumber = number;
      	pass2myteam = myteam;
      }


#if 0
      if((ipos.x > bestipos.x) && (myteam == true)){
	bestipos = ipos;
	bestdir = testdir[j];
	bestspeed = testspeed[j];
	bestadvantage = tmp_advantage;
	bestnumber = number;
	pass2myteam = myteam;
      }
#endif

    }
  } // for directions j
  
  dir = bestdir;
  speed = bestspeed;
  resulting_pos = bestipos;
  advantage = bestadvantage;
  closest_teammate = bestnumber;

  if(bestipos.getX() < -99.)
  {
    bestdir = 0.;
    speed = 0.;

    DBLOG_POL(0,<<"PT: NO Kicknrush found (all not in pitch), SETTING SPEED TO 0 "
	      <<RAD2DEG(bestdir)<<" speed "<<speed);
  }
  else
  {
    DBLOG_POL(0,<<"PT: Best Kicknrush "<<" dir "<<RAD2DEG(bestdir)<<" speed "<<speed
	      <<" intercept pos "<<bestipos<<" myteam intercepts "<<pass2myteam<<" player number "
	      <<bestnumber);
    DBLOG_POL(0,<<_2D<<C2D(bestipos.x,bestipos.y,1.0,"red"));
    DBLOG_POL(0,<<_2D<<L2D(ballpos.x,ballpos.y,bestipos.x,bestipos.y,"red"));
  }

  if(pass2myteam)
    return true; 
  else
    return false;
}

bool
Policy_Tools::isKicknRushInterceptionPointAcceptable( Vector ipos, 
                                                      int    advantage )
{
  return true;
}


bool Policy_Tools::get_best_clearance_kick(const Vector ballpos, const int testdirs,
					   const double testdir[], double testspeed[],
					   double &speed, double &dir){

  Vector bestipos = Vector(-100,0);
  bool pass2myteam = false;
  const float min_gain = 20;
  double bestdir; // default
  double bestspeed;
  if(testdirs >0){
    bestdir = testdir[0]; // default
    bestspeed = testspeed[0];
  }
  else{
    bestdir = 0;
    bestspeed = 2.5;
  }
  for(int j=0;j<testdirs;j++){
    Vector ipos;
    bool myteam;
    int number;
    if((testspeed[j]>2.5) || testspeed[j] <0){
      /*
      if(testspeed[j]>2.5) 
	cout<<"test speed "<< testspeed[j] <<" >2.5"<<endl;
      else if(testspeed[j] <0)
	cout<<"test speed "<< testspeed[j] <<" < 0"<<endl;
      */
      testspeed[j] = 2.5;
    }
    /*
    if(mdpInfo::is_opponent_in_kickrange()){
      testspeed[j] = Move_1Step_Kick::get_max_vel_in_dir(testdir[j]);
      //DBLOG_INFO("Test best clearance kick - op. in kickrange, dir/ speed "<<testdir[j]<<" "<<testspeed[j]);
    }	
    */

    bool inpitch=Policy_Tools::earliest_intercept_pos(ballpos,testspeed[j],testdir[j],
						      ipos, myteam, number);
#if 0
    DBLOG_POL(4,<<"PT: Testing Clearance "<<" dir "<<testdir[j]
	      <<" intercept pos "<<ipos<<" myteam intercepts "<<myteam<<" player number "
	      <<number);
#endif
    if(inpitch){
      if(pass2myteam == false){ // I haven't found a possibility to pass to my team member yet
	if((ipos.getX() > bestipos.getX()) || (myteam == true)){
	  bestipos = ipos;
	  bestdir = testdir[j];
	  bestspeed = testspeed[j];
	  pass2myteam = myteam;
	}
      }
      else{// the best position already goes to my player, so try to improve if possible
	if((ipos.getX() > bestipos.getX()) && (myteam == true)){
	  bestipos = ipos;
	  bestdir = testdir[j];
	  bestspeed = testspeed[j];
	  pass2myteam = myteam;
	}
      }
    } // ball is not in pitch, so continue;
  } // for directions j
  dir = bestdir;
  speed = bestspeed;
  if(pass2myteam || ((bestipos - ballpos).norm() > min_gain))
    return true; 
  else
    return false;
}

/* D
float Policy_Tools::get_goalshot(float & res_vel, Vector & res_target, float & res_dir,
				 const Vector & ball_pos){

  float test_vel[3];
  Vector test_target[5];
  float best_sl = -10;
  Vector best_target = Vector(0);
  float best_vel =0;
  float sl;

  if(ball_pos.x < 10)  // do not test if ballpos not more than 10m in opponents half
    return -1;

  test_vel[0] = 0;
  test_vel[1] = 2.5;
  int no_test_vel =2;
  
  float width = ServerOptions::goal_width/2.;
  double dist,tolerance;
  test_target[0] = mdpInfo::opponent_goalpos();
  
  //test_target[1] = mdpInfo::opponent_goalpos()+ Vector(0,+width-goalshot_corner_tolerance);
  //test_target[2] = mdpInfo::opponent_goalpos()+ Vector(0,-width+goalshot_corner_tolerance);
  
  test_target[1] = mdpInfo::opponent_goalpos()+ Vector(0,+width); // right corner
  dist = (WSinfo::me->pos - test_target[1]).norm();
  if(goalshot_corner_tolerance < .5)
    goalshot_corner_tolerance =.5;
  if(dist >17)
    tolerance = goalshot_corner_tolerance;
  else if(dist <5)
    tolerance = .5; // minimum tolerance for close distances
  else
    tolerance = (dist -5)/12. * (goalshot_corner_tolerance -.5) + .5;
  test_target[1].y -= tolerance;
  test_target[2] = mdpInfo::opponent_goalpos()+ Vector(0,-width); // left corner
  dist = (WSinfo::me->pos - test_target[2]).norm();
  if(dist >17)
    tolerance = goalshot_corner_tolerance;
  else if(dist <5)
    tolerance = .5; // minimum tolerance for close distances
  else
    tolerance = (dist -5)/12. * (goalshot_corner_tolerance-.5) + .5;
  test_target[2].y += tolerance;


  int no_test_target =3;

  for(int i=0;i<no_test_vel;i++){
    if(i>0 && (mdpInfo::is_opponent_in_kickrange()))
      // no chance to make 'mega' kick'
      continue;
    if(best_sl >0) // already found a kick with low velocity that scores
      continue;
    for(int j=0;j<no_test_target;j++){
      if(i==0)
	test_vel[i] = Tools::min(ServerOptions::ball_speed_max,Move_1Step_Kick::get_max_vel_in_dir((test_target[j] - ball_pos).arg()));
      if((sl=Policy_Tools::goalshot_success_level(ball_pos,test_target[j],test_vel[i]))>best_sl){
	best_sl = sl;
	best_target = test_target[j];
	best_vel = test_vel[i];
	
	//if((i==0) && (sl >0))
	//  DBLOG_INFO("policy_tools: Score with single kick should do the job! vel "
	//	   <<best_vel<<" target "<<best_target);
	
      }
    }
  }    
  
  if(best_sl>0){
    //DBLOG_POLICY_D<<"Get Goalshot best_vel "<<best_vel<<" Best target "<<best_target;
    //DBLOG_INFO("Get Goalshot best_vel "<<best_vel<<" Best target "<<best_target);
    res_vel = best_vel;
    res_target = best_target;
    res_dir = (best_target - ball_pos).arg();
    return best_sl;
  }
  return -1;
}
*/

/* D
float Policy_Tools::goalshot_success_level(Vector ballpos,  Vector target, float vel, 
					   bool worst_case ){
  
#define MAX_SCORE_ANGLE 80/180. *PI

  Vector ballvel;
  bool shoot2corner = false;
  float best_time = 1000;
  
  double dir = (target - ballpos).arg();
 
  if(Tools::get_abs_angle(dir)>=MAX_SCORE_ANGLE) // that's nearly impossible
    return -1.;

  if((target.y  >= ServerOptions::goal_width/2. - 2.) || //left corner 
     (target.y  <= -ServerOptions::goal_width/2. + 2.)) //right corner
    shoot2corner = true;

  ballvel.init_polar((double)vel,(Angle)dir);

  Vector max_ball_pos = ballvel;
  // compute the position where the ball stops if no opponent gets it
  //   subtract an offset of -.5 in order to avoid too weak shots 
  //max_ball_pos.normalize((vel-0.5) * 1./(1 - ServerOptions::ball_decay));  
  max_ball_pos.normalize(vel * 1./(1 - ServerOptions::ball_decay));  
  if(((ballpos + max_ball_pos).x - ServerOptions::pitch_length/2.) < 0)  // no chance to reach a goal yet
    return -1.;

  int goalie = mdpInfo::opponent_goalie_number();
  for(int op_idx = 0;op_idx<11;op_idx++){
    if(mdpInfo::mdp->his_team[op_idx].pos_x.p == 0)  // >0: position valid
      continue;
    Vector op_pos = mdpInfo::mdp->his_team[op_idx].pos();
    double angle2op = (op_pos - ballpos).arg();
    if(Tools::get_abs_angle(angle2op - ballvel.arg()) > 80./180. * PI){ // op. not in balldir
      if(mdpInfo::mdp->his_team[op_idx].number != goalie)
	continue; // do not consider
    }
    double dist2op = (mdpInfo::mdp->his_team[op_idx].pos()-ballpos).norm();
    if (vel > dist2op){
      if(mdpInfo::mdp->his_team[op_idx].number != goalie)
	continue;
    }
    Vector intercept_pos;
    int time;
    float player_size = 0;
    double player_speed=  ServerOptions::maxpower * ServerOptions::dash_power_rate;
    player_speed =  0.5 * player_speed + 0.5 * ServerOptions::player_speed_max;

    if(mdpInfo::mdp->his_team[op_idx].number == goalie){
      if(worst_case == false){ //this is the normal case
	player_size = goalshot_goalie_bonusstep;
	player_size += goalshot_consider_age * 
	  mdpInfo::age_playerpos(&(mdpInfo::mdp->his_team[op_idx]));
	if(player_size > CONSIDER_AGE_MAXINCREASE + goalshot_goalie_bonusstep){
	  player_size = CONSIDER_AGE_MAXINCREASE + goalshot_goalie_bonusstep;
	  //DBLOG_POLICY<<"Planning.c: Warning: age increases goalshot bonus too much. Restricting";
	}
	if(shoot2corner){
	  player_speed *= extremeshot_goalie_maxspeed_percentage;
	  player_size *= extremeshot_goalie_playersize_percentage;
	  //player_size -= goalshot_corner_tolerance;
	}
      }
      else{ // worst case consideration: goalie-size == player_age
	player_size = goalshot_goalie_bonusstep;
	player_size += goalshot_worstcase_consider_age * 
	  mdpInfo::age_playerpos(&(mdpInfo::mdp->his_team[op_idx]));
	if(shoot2corner){
	  player_speed *= extremeshot_goalie_maxspeed_percentage;
	}
      }
    }

    intercept_min_time_and_pos_hetero(time, intercept_pos, ballpos, 
			       ballvel,op_pos, mdpInfo::mdp->his_team[op_idx].number, false, 
			       player_speed, player_size);
    if(intercept_pos.x  - ServerOptions::pitch_length/2.<= 2.) 
      return -1; // someone gets the ball before the line
    if(time < best_time){
      best_time = time;
    }
  }
  return best_time;
}
*/


void Policy_Tools::intercept_player(int & res_time, Vector & res_pos,
					 const Vector A_pos, 
					 const double A_speed,
					 const double A_dir,
					 const Vector B_pos,
					 double player_vel_max,
				    double player_size){

  Intercept I;
  I.ball_pos = A_pos;
  Vector vel;
  vel.init_polar(A_speed,A_dir);
  I.ball_vel = vel;
  I.ball_vel_decay=  .999;  // assume no decaying of player A

  I.player_pos =    B_pos;
  if(player_size==-1000.0) // default radius for player
    I.player_radius=   ServerOptions::player_size;
  else
    I.player_radius=   player_size;
  if (player_vel_max < 0.0) { //default value for player_vel_max
    I.player_vel_max=  ServerOptions::maxpower * ServerOptions::dash_power_rate;
    I.player_vel_max=  0.5 * I.player_vel_max + 0.5 * ServerOptions::player_speed_max;
  }
  else //set custom's player_vel_max
    I.player_vel_max= player_vel_max;
  //DBLOG_POLICY<<"Player intercept A "<<A_pos<<"B "<<B_pos <<" vel "<<vel<<" Bvel "<<I.player_vel_max;
  I.minimal_time_player_reaches_ball_at_pos(res_time,res_pos);


}


void Policy_Tools::intercept_min_time_and_pos_hetero(int & res_time, Vector & res_pos,
						     const Vector & ball_pos, 
						     const double ballspeed,
						     const double balldir,
						     const Vector & player_pos, 
						     int player_number,
						     bool my_team,
						     double player_vel_max_factor,
						     double player_size_factor) {
  Vector ball_vel;

  ball_vel.init_polar(ballspeed,balldir);

  intercept_min_time_and_pos_hetero(res_time, res_pos, ball_pos, ball_vel,
			     player_pos, player_number, my_team, player_vel_max_factor, player_size_factor);
}


void Policy_Tools::intercept_min_time_and_pos_hetero(int & res_time, Vector & res_pos,
						     const Vector & ball_pos, 
						     const Vector & ball_vel, 
						     const Vector & player_pos, 
						     int player_number,
						     bool my_team,
						     double player_vel_max_factor,
						     double player_size_factor) {
  Intercept I;
  I.ball_pos = ball_pos;
  I.ball_vel = ball_vel;
  I.ball_vel_decay=  ServerOptions::ball_decay;
  I.player_pos =     player_pos;

  PPlayer player = NULL;

  if (player_number == -1) {
    player = NULL;
  } else {
    if (my_team) {
      player = WSinfo::get_teammate_by_number(player_number);
    } else {
      player = WSinfo::get_opponent_by_number(player_number);
    }
  }
  I.player = player;

  // ridi: quick hack to get old intercept behaviour
  if(use_go2ball_classic == true)
    player = NULL;

  double radius = ServerOptions::kickable_area;
  if (player != NULL) radius = player->kick_radius;

  if (player_size_factor < -0.1) {// default radius for player
    I.player_radius = radius;
  } else {
    I.player_radius = player_size_factor * radius;
  }

  double v_max = 0.9 * ServerOptions::player_speed_max;
  if (player != NULL) v_max = 0.9 * player->speed_max;
    
  if (player_vel_max_factor < 0.0) { //default value for player_vel_max
    //I.player_vel_max=  ServerOptions::maxpower * ServerOptions::dash_power_rate;
    //I.player_vel_max=  0.5 * I.player_vel_max + 0.5 * ServerOptions::player_speed_max;
    I.player_vel_max = v_max;
  } else {//set custom's player_vel_max
    I.player_vel_max= player_vel_max_factor * v_max;
  }

  I.minimal_time_player_reaches_ball_at_pos(res_time,res_pos);

}


/**********************************************************************/

/* Martin's intercept routines April 2002 */

/**********************************************************************/

int Policy_Tools::get_time2intercept(Vector & interceptpos,
				     const Vector & ballpos, 
				     const double ballspeed,
				     const double balldir,
				     const Vector & playerpos, 
				     const int time2react,
				     int max_steps ,
				     const bool goalie ,
				     const float kickrange_percentage ){
  Vector ballvel;
  ballvel.init_polar(ballspeed,balldir);

  return get_time2intercept(interceptpos,ballpos,ballvel,playerpos,
			    time2react,max_steps,goalie);
}

int Policy_Tools::get_time2intercept(Vector & interceptpos,
				     const Vector & ballpos, 
				     const Vector initballvel, 
				     const Vector & playerpos, 
				     const int time2react,
				     int max_steps ,
				     const bool goalie,
				     const float kickrange_percentage ){

  /** time2react: default is 1: in cycle 1 after kick, the player sees the ball
      and starts to move.
      therefore: time2react=0 means the player is already moving towards the ball
      time2react=2: the player additionally has to turn
  */

  if (max_steps > MAX_STEPS2INTERCEPT)
    max_steps = MAX_STEPS2INTERCEPT;

  Vector ballpos_at_t = ballpos;
  Vector ballvel = initballvel;
  float intercept_radius = 0.0;
  float intercept_range;
  /* if the ball is in kickrange of player at time 0, wait until ball left kickr: */
  bool ball_hasleft_kickrange = false; 

  if((ballpos - playerpos).sqr_norm() > SQUARE(ServerOptions::kickable_area))
    ball_hasleft_kickrange = true;
  
  if ((goalie == true) 
      &&  RIGHT_PENALTY_AREA.inside(ballpos_at_t) ) 
    intercept_range = ServerOptions::catchable_area_l;
  else
    intercept_range = ServerOptions::kickable_area * kickrange_percentage;

  int time = 0;
  while (time <=max_steps) {
    //  TIME      0      1            2      3     4
    //  Player     turn   turn         dash    0.4   0.6  
    //  BALL   kick initVel*decay   ...
    ballpos_at_t += ballvel;
    ballvel *= ServerOptions::ball_decay;
    time++;

    if (time < time2react +1)
      intercept_radius = 0.0;
    else if (time < time2react + 2)
      intercept_radius = 0.6;
    else if (time < time2react + 3)
      intercept_radius = 0.6 + 0.8;
    else 
      intercept_radius += .93*ServerOptions::player_speed_max;

    //    if((ballpos_at_t-playerpos).norm() < (intercept_radius+intercept_range)){
    //    if((ballpos_at_t-playerpos).sqr_norm() < (intercept_radius+intercept_range) *(intercept_radius+intercept_range)){
    if((ballpos_at_t-playerpos).sqr_norm() < SQUARE(intercept_radius+intercept_range)){
      if(ball_hasleft_kickrange == true){ 
	// player got ball!
	interceptpos = ballpos_at_t;
	return time;
      }
    }

    if((ballpos_at_t - playerpos).sqr_norm() > SQUARE(ServerOptions::kickable_area))
      ball_hasleft_kickrange = true;
  }
  return -1; // player doesn't get ball before max time is over
}


int Policy_Tools::get_time2intercept_hetero(Vector & interceptpos,
					    const Vector & ballpos, 
				     const double ballspeed,
				     const double balldir,
					    const PPlayer & player, 
					    const int time2react,
					    int max_steps ,
					    const bool goalie,
					    const float kickrange_percentage ){
  Vector ballvel;
  ballvel.init_polar(ballspeed,balldir);

  return get_time2intercept_hetero(interceptpos,ballpos,ballvel,player,
				   time2react,max_steps,goalie);

}

int Policy_Tools::get_time2intercept_hetero(Vector & interceptpos,
					    const Vector & ballpos, 
					    const Vector initballvel, 
					    const PPlayer & player, 
					    const int time2react,
					    int max_steps ,
					    const bool goalie,
					    const float kickrange_percentage ){

  /** time2react: default is 1: in cycle 1 after kick, the player sees the ball
      and starts to move.
      therefore: time2react=0 means the player is already moving towards the ball
      time2react=2: the player additionally has to turn
  */

  if(player == NULL)
    return -1;

  if (max_steps > MAX_STEPS2INTERCEPT)
    max_steps = MAX_STEPS2INTERCEPT;

  Vector ballpos_at_t = ballpos;
  Vector ballvel = initballvel;
  float intercept_radius = 0.0;
  float intercept_range;
  /* if the ball is in kickrange of player at time 0, wait until ball left kickr: */
  bool ball_hasleft_kickrange = false; 

  Vector playerpos = player->pos;

  if(player->age_vel <= 1){ // I saw the velocity of that player, so consider it for the new position
    Vector vel = player->vel;
    Vector playerpos1 = playerpos + vel;  // this will happen in any case.
    vel *= (1/ player->decay); // this would be only true, if player actively dashes; this is considered later.
    Vector playerpos2 = playerpos + vel;
    double dist1 = Tools::get_dist2_line(ballpos, ballpos + 10*initballvel,playerpos1);
    double dist2 = Tools::get_dist2_line(ballpos, ballpos + 10*initballvel,playerpos2);

    if(dist1 < 10.0 && dist2 < dist1){ // active dashing gets player closer to intercept line
      playerpos = playerpos2;
#if 0
      LOG_POL(0,"time2intercept_hetero: Player "<<player->number
	      <<" vel age is <=1. Player approaches intercept line. Adapting playerpos. Was: "<<player->pos
	      <<" is: "<<playerpos<<" vel: "<<player->vel<<" corrected: "<<vel<<" dist2line "<<dist2);
#endif
    }
    else 
      playerpos = playerpos1;
  }


  if((ballpos - playerpos).sqr_norm() > SQUARE(player->kick_radius))
    ball_hasleft_kickrange = true;
  
  if ((goalie == true) 
      &&  RIGHT_PENALTY_AREA.inside(ballpos_at_t) ) 
    intercept_range = ServerOptions::catchable_area_l;
  else
    intercept_range = player->kick_radius * kickrange_percentage;

  //  LOG_POL(0,<<"checking time2intercept for "<<player->number<<" speed "<<player->speed_max);


  int time = 0;
  while (time <=max_steps) {
    //  TIME      0      1            2      3     4
    //  Player     turn   turn         dash    0.4   0.6  
    //  BALL   kick initVel*decay   ...
    ballpos_at_t += ballvel;
    ballvel *= ServerOptions::ball_decay;
    time++;

    if (time < time2react +1)
      intercept_radius = 0.0;
    else if (time < time2react + 2)
      intercept_radius = Tools::speed_after_n_cycles(1,player->dash_power_rate,
						     player->effort, player->decay);
    else if (time < time2react + 3)
      intercept_radius = Tools::speed_after_n_cycles(2,player->dash_power_rate,
						     player->effort, player->decay);
    else if (time < time2react + 4)
      intercept_radius = Tools::speed_after_n_cycles(3,player->dash_power_rate,
						     player->effort, player->decay);
    else
      intercept_radius += player->speed_max;

    if((ballpos_at_t-playerpos).sqr_norm() < SQUARE(intercept_radius+intercept_range)){
      if(ball_hasleft_kickrange == true){ 
	// player got ball!
	interceptpos = ballpos_at_t;
	return time;
      }
    }

    if((ballpos_at_t - playerpos).sqr_norm() > SQUARE(player->kick_radius))
      ball_hasleft_kickrange = true;
  }
  return -1; // player doesn't get ball before max time is over
}

int Policy_Tools::get_selfpass_advantage(const double speed, const double dir, Vector &interceptpos){
  int optime2react = 1;
  int mytime2react = 1;

  if(WSinfo::me->ang.diff(ANGLE(dir)) <10/180.*PI){
    mytime2react --;
  }

  int myteam_steps = get_time2intercept_hetero(interceptpos, WSinfo::ball->pos, 
					       speed,dir, WSinfo::me,
					       mytime2react); 
  if(myteam_steps <0 || Tools::is_position_in_pitch(interceptpos, 5) == false)
    return -100;  // I dont get the ball

  Vector op_ipos;
  int opteam_steps = opteam_time2intercept(WSinfo::ball->pos,speed,dir,op_ipos,optime2react,0,true);

  return opteam_steps - myteam_steps;

}


int Policy_Tools::opteam_time2intercept(const Vector ballpos,const double speed,
					const double dir,
					Vector &interceptpos,
					const int op_time2react ,
					const  int critical_value ,
					const  bool consider_opdir,
					const Vector centre_of_interest ,
					const float radius_of_interest ,
					const int max_age){

  // if we find an opponent that beats the critical_value, we're done !
  // check opponents

  int op_fastest = 2000;

  PlayerSet pset = WSinfo::valid_opponents;

  for (int idx = 0; idx < pset.num; idx++) {
    //    Vector op_pos;
    if(pset[idx]->age >max_age){
      //LOG_POL(0,"Checking pass: do not consider opponent "<<pset[idx]->number<<" too old: "<<pset[idx]->age);
      continue;
    }
    bool is_goalie;
          
    if(pset[idx]->number ==mdpInfo::mdp->his_goalie_number)
      is_goalie=true;
    else
      is_goalie=false;
    
    if((pset[idx]->pos - centre_of_interest).sqr_norm() > SQUARE(radius_of_interest))
      continue;


    Vector player_intercept_pos;
    const float opteam_kickrange_percentage = 1.0;
    int steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
					  speed,dir, pset[idx] ,op_time2react,
					  op_fastest,
					  is_goalie,opteam_kickrange_percentage); 
    if(consider_opdir == true){
      if(pset[idx]->ang.diff((player_intercept_pos-pset[idx]->pos).ARG()) >30/180.*PI){
	steps ++;
      }
      else{
	steps --;
      }
    }

    if(is_goalie == true && RIGHT_PENALTY_AREA.inside(player_intercept_pos) == false){
      // LOG_POL(1,"Goalie will get the pass outside penalty area -> do not consider !");
      steps = 2000;
    }

    if(steps <0) 
      steps =2000; // player doesnt get ball in reasonable time
    if(steps < op_fastest){ // the fastest yet
      op_fastest = steps;
      if(op_fastest <= critical_value){
	// opponent is as fast or faster at ball -> done!
	interceptpos = player_intercept_pos;
	break; // stop search
      }
    }
  }
  return op_fastest;
}

bool Policy_Tools::myteam_intercepts_ball_hetero(const Vector ballpos,const double speed,
						 const double dir,
						 Vector &intercept_pos,
						 int &advantage,
						 int &number,Vector &playerpos,
						 const double myteam_time2react,
						 const double opteam_time2react){




  int myteam_fastest = INFINITE_STEPS;
  Vector earliest_intercept = Vector(0);

  // check my team
  int playerage = 1000;

  PlayerSet pset = WSinfo::valid_teammates;
  for (int idx = 0; idx < pset.num; idx++) {
    int my_time2react;
    if(myteam_time2react == 1000)
      my_time2react = pass_mytime2react;
    else
      my_time2react = myteam_time2react;

    if(pset[idx]->number == WSinfo::me->number){ // check selfpass
      if(Tools::get_abs_angle(Tools::get_angle_between_null_2PI(dir - 
								WSinfo::me->ang.get_value()))
	 >4./180. *PI){
	//DBLOG_POL(BASELEVEL+0,<<"SelfLaufpass in dir "<<RAD2DEG(dir)<<" I have to turn first ");
	my_time2react ++;
      }
    }

    Vector player_intercept_pos;
    int steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
					  speed,dir, pset[idx],
					  my_time2react,myteam_fastest); 
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      number = pset[idx]->number;
      earliest_intercept = player_intercept_pos;
      playerpos=pset[idx]->pos;
      playerage =pset[idx]->age;
    }
  }
  intercept_pos = earliest_intercept;

  if(myteam_fastest>=INFINITE_STEPS)
    return false;
  
  // here we now: myteam_fastest
  // if we find an opponent that beats that, we're done
  // check opponents

  Vector op_ipos;

  // ridi05: considered harmful:  int op_maxage = playerage +5;
  int op_maxage = playerage +100;

  int op_time2react;
  if(opteam_time2react == 1000)
    op_time2react = pass_optime2react;
  else
    op_time2react = opteam_time2react;

  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, op_time2react, 
				       myteam_fastest,false,intercept_pos, 
				       (intercept_pos- ballpos).norm() * ServerOptions::player_speed_max+IGNORE_OPS_SAFETY, op_maxage);

  DBLOG_POL(0,"PT: check  myteamintercept_hetero "<<number<<" gets ball after "<<myteam_fastest
	  <<" op needs "<<op_fastest);

  advantage = op_fastest - myteam_fastest;
  if(advantage >= pass_my_minadvantage){
    // our team is faster than opponent; if pos in pitch, everything's fine
    if(mdpInfo::is_position_in_pitch(earliest_intercept,2.0) == true)
      return true;
    advantage = -1;
    return false;
  }
  else{ // opponent is faster or equally fast
    intercept_pos = op_ipos;
    return false;
  }
}

bool Policy_Tools::compute_ipos_for_teammate06(const Vector ballpos,const double speed, const double dir, const int  teammate,
					       int &num_steps, Vector & intercept_pos, const int my_time2react){

  Vector tmp_ballpos = ballpos;

  num_steps = -1;
  InterceptResult result;
  Vector ballvel;
  ballvel.init_polar(speed,dir);
  //  LOG_POL(0,<<"policy_tools: my_time2react: "<<my_time2react);
  for(int i= 0; i<my_time2react;i++){
    tmp_ballpos += ballvel;
    ballvel *= ServerOptions::ball_decay; //ball
  }

  // ridi 06: try this to estimate the ipos more realistically.
#if 0  // ridi: do NOT activate: with (*) too cautious: without (*) its too risky
  tmp_ballpos += ballvel;
  ballvel *= ServerOptions::ball_decay; //ball
#endif

  PlayerSet pset1;
  pset1.append(WSinfo::valid_teammates.get_player_by_number(teammate));
  pset1.keep_and_sort_best_interceptors_with_intercept_behavior(1, tmp_ballpos, ballvel, &result);
  if(pset1.num!= 0){
    num_steps = result.time;
    num_steps += my_time2react;
#if 0 // (*) comment see above
    num_steps ++;
#endif
    intercept_pos = result.pos;
    return true;
  }
  return false;
}




bool Policy_Tools::myteam_intercepts_ball_hetero06(const Vector ballpos,const double speed,
						   const double dir,
						   Vector &intercept_pos,
						   int &advantage,
						   int &number,Vector &playerpos,
						   const double myteam_time2react,
						   const double opteam_time2react,
						   const int desired_receiver){




  int myteam_fastest = INFINITE_STEPS;
  Vector earliest_intercept = Vector(0);

  // check my team
  int my_time2react;
  if(myteam_time2react == 1000)
    my_time2react = pass_mytime2react;
  else
    my_time2react = myteam_time2react;

  number = desired_receiver;

  PlayerSet pset = WSinfo::valid_teammates;
  for (int idx = 0; idx < pset.num; idx++) {
    if(desired_receiver >0){ // there's a specific receiver specified
      if(pset[idx]->number != desired_receiver) 
	continue;
    }
    
    if(pset[idx]->number == WSinfo::me->number  && desired_receiver <0){ // do not consider me, if not explicitly specified
      continue;  // do not consider me!
    }
    
    Vector player_intercept_pos;
    int steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
					  speed,dir, pset[idx],
					  my_time2react,myteam_fastest); 
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      number = pset[idx]->number;
      earliest_intercept = player_intercept_pos;
      playerpos=pset[idx]->pos;
    }
  }
  intercept_pos = earliest_intercept;
  
  if(myteam_fastest>=INFINITE_STEPS)
    return false;

  Vector ballvel;
  ballvel.init_polar(speed,dir);

  PPlayer teammate =WSinfo::valid_teammates.get_player_by_number(number);
  double distance = Tools::get_dist2_line(ballpos, (ballpos + 30*ballvel),
					 teammate->pos);
  if(distance > teammate->kick_radius){
    //LOG_POL(0,"pass in dir "<<RAD2DEG(dir)<<" is a NOT direct pass. Exact ipos computation ");
    // do this computation only for none direct passes
    Vector ipos;  // result of compute_ipos...
    int num_steps; // result of compute_ipos...
  
    if(compute_ipos_for_teammate06(ballpos, speed, dir, number, num_steps, ipos, my_time2react) == true){
      intercept_pos = ipos;
      myteam_fastest = num_steps;
    }
  }
  else{
    //    LOG_POL(0,"pass in dir "<<RAD2DEG(dir)<<" is a direct pass. Simple ipos computation. mysteps: "<<myteam_fastest);
    ;
  }

  // here we know: myteam_fastest
  // if we find an opponent that beats that, we're done
  // check opponents

  Vector op_ipos;

  // ridi05: considered harmful:  int op_maxage = playerage +5;
  int op_maxage = 1000;

  int op_time2react;
  if(opteam_time2react == 1000)
    op_time2react = pass_optime2react;
  else
    op_time2react = opteam_time2react;

  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, op_time2react, 
				       myteam_fastest,false,intercept_pos, 
				       (intercept_pos- ballpos).norm() * ServerOptions::player_speed_max+IGNORE_OPS_SAFETY, op_maxage);


  //LOG_POL(0,"PT: check  myteamintercept_hetero "<<number<<" gets ball after "<<myteam_fastest  <<" op needs "<<op_fastest);

  advantage = op_fastest - myteam_fastest;
  if(advantage >= pass_my_minadvantage){
    // our team is faster than opponent; if pos in pitch, everything's fine
    if(mdpInfo::is_position_in_pitch(intercept_pos,2.0) == true)
      return true;
    advantage = -1;
    return false;
  }
  else{ // opponent is faster or equally fast
    intercept_pos = op_ipos;
    return false;
  }
}





bool Policy_Tools::opteam_intercepts_ball_faster06(const Vector ballpos,const double speed,
						   const double dir,
						   const int my_num_steps,
						   const Vector my_ipos,
						   int & advantage,
						   const double opteam_time2react){

  // here we know: myteam_fastest
  // if we find an opponent that beats that, we're done
  // check opponents

  Vector op_ipos;
  int op_maxage = 100;

  int op_time2react;
  if(opteam_time2react == 1000)
    op_time2react = pass_optime2react;
  else
    op_time2react = opteam_time2react;
  
   int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, op_time2react, 
				       my_num_steps,false,my_ipos, 
				       (my_ipos- ballpos).norm() * ServerOptions::player_speed_max
					+IGNORE_OPS_SAFETY, op_maxage);

  advantage = op_fastest - my_num_steps;
  if(advantage >= pass_my_minadvantage){
    return true;
  }
  return false;
}














bool Policy_Tools::myplayer_intercepts_ball(const Vector ballpos,const double speed,
					    const double dir,
					    const Vector playerpos,
					    Vector &intercept_pos,
					    int &advantage){
  
  int my_time2react = pass_mytime2react;
  Vector player_intercept_pos;
  int myteam_steps = get_time2intercept(player_intercept_pos, ballpos, 
					speed,dir, playerpos,
					my_time2react); 
  if(myteam_steps <0) 
    return false;
  intercept_pos =  player_intercept_pos;

  Vector op_ipos;
  
  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, pass_optime2react, 
				       myteam_steps,false,player_intercept_pos, 
				       (player_intercept_pos- ballpos).norm() * ServerOptions::player_speed_max+IGNORE_OPS_SAFETY);

  advantage = op_fastest - myteam_steps;
  if(advantage >= pass_my_minadvantage){
    // our team is faster than opponent; if pos in pitch, everything's fine
    if(mdpInfo::is_position_in_pitch(intercept_pos,2.0) == true)
      return true;
    advantage = -1;
    return false;
  }
  else{ // opponent is faster or equally fast
    intercept_pos = op_ipos;
    return false;
  }
}

bool Policy_Tools::myplayer_intercepts_ball_hetero(const Vector ballpos,const double speed,
						   const double dir,
						   const PPlayer player,
						   Vector &intercept_pos,
						   int &advantage){
  
  int my_time2react = pass_mytime2react;
  Vector player_intercept_pos;
  int myteam_steps = get_time2intercept_hetero(player_intercept_pos, ballpos, 
					       speed,dir, player,
					       my_time2react); 
  if(myteam_steps <0) 
    return false;
  intercept_pos =  player_intercept_pos;

  Vector op_ipos;
  
  int op_maxage = 1000;
  if(player)
    op_maxage = player->age+5;

  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, pass_optime2react, 
				       myteam_steps,false,player_intercept_pos, 
				       (player_intercept_pos- ballpos).norm() *ServerOptions::player_speed_max+IGNORE_OPS_SAFETY, op_maxage);

  DBLOG_POL(0,"PT: check  pass2myplayer_hetero "<<player->number<<" gets ball after "<<myteam_steps
	  <<" max_speed: "<<player->speed_max
	  <<" op needs "<<op_fastest);

  advantage = op_fastest - myteam_steps;
  if(advantage >= pass_my_minadvantage){
    // our team is faster than opponent; if pos in pitch, everything's fine
    if(mdpInfo::is_position_in_pitch(intercept_pos,2.0) == true)
      return true;
    advantage = -1;
    return false;
  }
  else{ // opponent is faster or equally fast
    intercept_pos = op_ipos;
    return false;
  }
}

bool Policy_Tools::I_am_fastest4pass(const Vector ballpos,const double speed,
				     const double dir,
				     Vector &intercept_pos){
  //added my_num in order to avoid that noone is going to the ball,
  //thus taking the risk that two players might go there.
  int my_num=-2;
  if(WSinfo::me->number<6 && WSinfo::ball->pos.getX()<-33)
    my_num=WSinfo::me->number;
  
  int myteam_fastest = INFINITE_STEPS;
  Vector earliest_intercept = Vector(0);
  int number = 0;

  // check my team
  for(int idx = 0;idx<11;idx++){
    if(mdpInfo::mdp->my_team[idx].pos_x.p == 0)  // >0: position valid
      continue;
    Vector player_intercept_pos;
    int steps = get_time2intercept(player_intercept_pos, ballpos, 
				   speed,dir, mdpInfo::mdp->my_team[idx].pos(),
				   2,myteam_fastest); 

    LOG_POL(BASELEVEL,<<"policy_tools: Player "<<mdpInfo::mdp->my_team[idx].number<<" after "<<steps<<" at "
	    <<player_intercept_pos);
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(idx==my_num-1)
      steps--;
    if(steps==myteam_fastest && number==my_num && idx<5)
      continue;
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      number = mdpInfo::mdp->my_team[idx].number;
      earliest_intercept = player_intercept_pos;
      //playerpos=mdpInfo::mdp->my_team[idx].pos();
    }
  }

  LOG_POL(BASELEVEL,<<"policy_tools: Fastest Player "<<number<<" after "<<myteam_fastest<<" at "
	  <<earliest_intercept);
  intercept_pos = earliest_intercept;
  LOG_POL(BASELEVEL,<<_2D<<VC2D(intercept_pos,1.0,"yellow"));
  if(myteam_fastest<=INFINITE_STEPS && number==mdpInfo::mdp->me->number)
    return true;
  return false;
}  
  // here we now: myteam_fastest
  // if we find an opponent that beats that, we're done
  // check opponents


bool Policy_Tools::check_go4pass(Vector &ipos){
  Vector ballpos,ballvel;
  return check_go4pass(ipos,ballpos,ballvel);
}


bool Policy_Tools::check_go4pass(Vector &ipos, Vector &ballpos, Vector &ballvel){
  //  Vector ballpos;
  
  //added my_num in order to avoid that noone is going to the ball,
  //thus taking the risk that two players might go there.
  double speed;
  double dir;
  bool result;
  // check if someone communicated pass intention
#if 1
  
  PPlayer p= WSinfo::get_teammate_with_newest_pass_info();
  if ( p ) {
    speed = p->pass_info.ball_vel.norm();
    dir = p->pass_info.ball_vel.arg();
    ballpos = p->pass_info.ball_pos;
    ballvel = p->pass_info.ball_vel;
    LOG_POL(0, << "Policy_Tools: check go4pass: got pass info from player " 
	    << p->number  
	    << " a= " << p->pass_info.age
	    << " p= " << ballpos
	    << " v= " << p->pass_info.ball_vel
	    << " at= " << p->pass_info.abs_time
	    << " rt= " << p->pass_info.rel_time
	    <<" speed= "<<speed <<" dir "<<RAD2DEG(dir));
    result = I_am_fastest4pass(ballpos, speed, dir, ipos);
    return result;
  }
  else{
    LOG_POL(1, "Policy_Tools: check go4pass: no pass info");
    return false;
  }
#endif

}


/******************************************************************************
   Defense stuff (for questions ask DAVE!)
*******************************************************************************/

/* return array with numbers of opponent attackers to take care of 
   take into account  at most max_attacker opponents
*/
void Policy_Tools::get_opponent_attackers(int *attacker_number, int max_attackers, int op_ball_holder){
  /* (1) create an array with numbers all relevant opponent players
     and corresponding distances to my goal (opponent array) */
  double *dist2goal = new double[11];
  int *opponent = new int[11];
  
  for(int i=1;i<=11;i++){
    opponent[i-1] = -1;
    /* ignore opponents with invalid infos */
    if(!mdpInfo::is_opponent_pos_valid(i)) continue;
    /* ignore opponent holding the ball (its the interceptors job) */
    if(i == op_ball_holder) continue;
    Vector op_pos = mdpInfo::opponent_pos_abs(i);
    /* ignore opponents far away from our goal) */
    if(fabs(op_pos.getY())>15) continue;
    if(op_pos.getX() > -52.5+18) continue;
    if(op_pos.distance(Vector(-52.5,0)) > 30 ) continue;
    double goal_dist = op_pos.distance(Vector(-52.5,0));
    dist2goal[i-1] = goal_dist;
    opponent[i-1] = i; 
    //DBLOG_POLICY << "Found Opponent "<<  i << " Dist: " << goal_dist;
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
      if(dist2goal[i] < min_dist){
	min_dist = dist2goal[i];
	min_number_idx = i;
      }
    }

    /* found nearest attacker? */
    if(min_number_idx > -1){
      //DBLOG_POLICY << "Found Attacker "<<  opponent[min_number_idx];
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
  delete dist2goal;
  delete opponent;
}

#if 0
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/************************  T  E  S  T  ****************************************/
DebugOutput *GlobalLogger;
int main() {
  ServerOptions::init();  
  ServerOptions::read_from_file("../conf/server.conf");  
  int time;
  Vector pos;
  Policy_Tools::intercept_min_time_and_pos(time,pos,
					   Vector(5.0,0.0), Vector(ServerOptions::ball_speed_max,0.0), Vector(0.0,0.0), 1.0);
  cout << "\n Res_time= " << time << "  Res_pos= " << pos << endl;
  return 1;
}
#endif 






bool Policy_Tools::myteam_intercepts_ball(const Vector ballpos,const double speed,
					  const double dir,
					  Vector &intercept_pos,
					  int &advantage,
					  int &number,Vector &playerpos){
  int myteam_fastest = INFINITE_STEPS;
  Vector earliest_intercept = Vector(0);

  // check my team
  for(int idx = 0;idx<11;idx++){
    int my_time2react = pass_mytime2react;
    if(mdpInfo::mdp->my_team[idx].pos_x.p == 0)  // >0: position valid
      continue;
    if(mdpInfo::mdp->my_team[idx].number == mdpInfo::mdp->me->number){ // check selfpass
      if(Tools::get_abs_angle(Tools::get_angle_between_null_2PI(dir - 
								mdpInfo::mdp->me->ang.v))
	 >4./180. *PI){
	//DBLOG_POL(BASELEVEL+0,<<"SelfLaufpass in dir "<<RAD2DEG(dir)<<" I have to turn first ");
	my_time2react ++;
      }
    }

    Vector player_intercept_pos;
    int steps = get_time2intercept(player_intercept_pos, ballpos, 
				   speed,dir, mdpInfo::mdp->my_team[idx].pos(),
				   my_time2react,myteam_fastest); 
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      number = mdpInfo::mdp->my_team[idx].number;
      earliest_intercept = player_intercept_pos;
      playerpos=mdpInfo::mdp->my_team[idx].pos();
    }
  }
  intercept_pos = earliest_intercept;

  if(myteam_fastest>=INFINITE_STEPS)
    return false;
  
  // here we now: myteam_fastest
  // if we find an opponent that beats that, we're done
  // check opponents

  Vector op_ipos;
#if 0
  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, pass_optime2react, 
				       myteam_fastest);
#endif

  int op_fastest=opteam_time2intercept(ballpos,speed,dir, op_ipos, pass_optime2react, 
				       myteam_fastest,false,intercept_pos, 
				       (intercept_pos- ballpos).norm()+IGNORE_OPS_SAFETY);

#if 0
  if(op_fastest != op_time2ip && op_time2ip != INFINITE_STEPS)
    DBLOG_ERR(BASELEVEL,"MyTeamInterrcepts op_times differ by "<<op_fastest-op_time2ip
	    <<" op_time2intercept: "
	    <<op_fastest<<" op_time2ip "<<op_time2ip<<" ipos "<<intercept_pos
	    <<" my steps "<<myteam_fastest);

  //  DBLOG_POL(BASELEVEL,"op_time2intercept: "<<op_fastest<<" op_time2ip "<<op_time2ip);
#endif



  advantage = op_fastest - myteam_fastest;
  if(advantage >= pass_my_minadvantage){
    // our team is faster than opponent; if pos in pitch, everything's fine
    if(mdpInfo::is_position_in_pitch(earliest_intercept,2.0) == true)
      return true;
    advantage = -1;
    return false;
  }
  else{ // opponent is faster or equally fast
    intercept_pos = op_ipos;
    return false;
  }
}

