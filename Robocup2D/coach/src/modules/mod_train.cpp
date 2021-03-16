/* Author: Andreas Hoffmann
 *
 * This is a module for training with the async server
 *
 */

#include "mod_train.h"
#include "messages.h"
#include "str2val.h"

namespace sequence{
  long start_time;
  long duration;
  long loops;
  long near_goals;
  long goalie_catched;
  long last_protocolled_at;
}

namespace statistics{
  long lap_runs;
  long lap_cycles;
  long cycles;
  long sequence_len;
  long max_sequence_len;
  long total_time;
}

namespace parameter{
  bool stopif_op_ball;
  bool stopif_team_ball;
  bool stopif_ball_left;
  bool stopif_pass_success;
  bool stopif_goal;
  bool stopif_ball_intercepted;
  long stop_after_loops;
  long statfreq;
  bool autostart;
  float targetspeed, targetdir;
  float ballspeed_tolerance, balldir_tolerance;
  float kickrange;
  bool corner_kick_left;
  long stop_after_reseeds;
}
#define KICK_RANGE_DEF 1.085



bool ModTrain::init(int argc,char **argv) {
  cout << "\nInitialising Train-Module...";
  argc--;
  argv++;
  sit_loop= false;

  sprintf(prot_file,"%s","");

  ValueParser vp(argc,argv);
  vp.get("train_file", train_file, MAX_STR_LEN);
  vp.get("conf_file", conf_file, MAX_STR_LEN);
  vp.get("prot_file", prot_file, MAX_STR_LEN);

  protocol_on = false;

  if ( strlen(prot_file)>1 ) {
    cout<<"Protfile "<<prot_file<<endl;
    protf=fopen(prot_file,"w");
    protocol_on=true;
  }

  cout<<"Protocol: "<<protocol_on<<endl;

  if ( strlen(train_file) ) {
    cout << "\nLoading Table from " << train_file;
    sit.load_table(train_file);
  }
  // defaults:
  sequence::last_protocolled_at = -1;
  sequence::duration = 1000000; // default: don't stop
  statistics::lap_runs = 0;
  statistics::cycles = 0;
  statistics::lap_cycles = 0;
  statistics::total_time = 0;

  parameter::stopif_op_ball = false;
  parameter::stopif_goal = false;
  parameter::stopif_pass_success= false;
  parameter::stopif_ball_intercepted = false;
  parameter::stop_after_loops = 0;
  parameter::statfreq = 100;
  parameter::targetspeed = 2.5;
  parameter::targetdir = 0.0;
  parameter::ballspeed_tolerance = 0.2;
  parameter::balldir_tolerance = 0.07;
  parameter::autostart = false;
  parameter::kickrange = KICK_RANGE_DEF;
  parameter::corner_kick_left = false;
  parameter::stop_after_reseeds = 0;

  if ( strlen(conf_file) ) {
    ValueParser vp(conf_file);   
    vp.get("duration", sequence::duration);
    vp.get("stopif_op_ball", parameter::stopif_op_ball);
    vp.get("stopif_team_ball", parameter::stopif_team_ball);
    vp.get("stopif_ball_left", parameter::stopif_ball_left);
    vp.get("stopif_pass_success", parameter::stopif_pass_success);
    vp.get("stopif_ball_intercepted", parameter::stopif_ball_intercepted);
    vp.get("stopif_goal", parameter::stopif_goal);
    vp.get("stop_after_loops", parameter::stop_after_loops);
    vp.get("statfreq", parameter::statfreq);
    vp.get("targetspeed", parameter::targetspeed);
    vp.get("speed_tolerance", parameter::ballspeed_tolerance);
    vp.get("autostart", parameter::autostart);
    vp.get("kickrange", parameter::kickrange);
    vp.get("start_with_corner_kick_left", parameter::corner_kick_left);
    vp.get("stop_after_reseeds", parameter::stop_after_reseeds);
  }
  else{
    cout<<"\nTrainCoach: Warning: no conf-file defined"<<endl;
  }
  if(parameter::autostart==true){
    sit_loop= true;
    cout << "\nStarting first sequence!" << flush;
    start_new_sequence();
  }
  player_type = 0;
  return true;
}


bool ModTrain::destroy() {
  
  return true;
}


bool ModTrain::behave() {
  int stopped; long reseed_count=0;
  //cycle player types
  if(parameter::stop_after_reseeds==reseed_count) sit_loop = false;
  if(sequence::loops == parameter::stop_after_loops){
    protocol_sit();
    sequence::loops = 0;
    statistics::total_time = 0;
    player_type++;
    if(player_type>6 && parameter::stop_after_reseeds>=reseed_count){
      player_type = 1;
      cout << "\nReseeding Player Types!" << flush;
      reseed_count++;
      MSG::sendMsg(MSG::MSG_RESEED_HETRO);
    }
    if(player_type<7) MSG::sendMsg(MSG::MSG_CHANGE_PLAYER_TYPE, fld.getMyTeamName(), 1, player_type);
  }
  
  if(sit_loop){
    if(fld.getPM() != PM_play_on){
      MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_play_on);
    }
    if(parameter::corner_kick_left && sequence::start_time >= fld.getTime() - 2){
      MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_corner_kick_l);
    } else if(fld.getPM()!= PM_play_on){
      //      fld.setPM(PM_play_on);
    }
    statistics::lap_cycles ++;
    statistics::cycles ++;
    statistics::sequence_len++;
    statistics::total_time++;
    if((stopped = check_stop_sequence())){
      statistics::lap_cycles --;  //correct value
      statistics::sequence_len--; //correct value
      protocol_sit(stopped);
      if(statistics::sequence_len > statistics::max_sequence_len)
	statistics::max_sequence_len = statistics::sequence_len;
      statistics::sequence_len = 0;
      start_new_sequence(); // nach dem Spiel ist vor dem Spiel
      if(stopped == BALL_INTERCEPTED){
	
      }
    }
    else
      protocol_sit();
  }  
  return true;
}


bool ModTrain::onRefereeMessage(bool PMChange) {
  
  return true;
}


bool ModTrain::onKeyboardInput(const char *str) {
  // start loop
  if (! strncmp(str,"l",1)){
    int n=0;
    sscanf(str,"%*s %d",&n);
    if(n>0)
      parameter::stop_after_loops=n;
    sequence::loops = 0;
    sequence::near_goals = 0;
    sit_loop= true;
    start_new_sequence();
    return true;
  }
  // stop loop
  if (! strncmp(str,"!l",2)){
    sit_loop= false;
    cout << "\n>>> loop off " << flush;
    return true;
  }
  // set situation
  if (! strncmp(str,"s",1)){
    int n;
    sscanf(str,"%*s %d",&n);
    sit.set_cur_sit(n);
    start_new_sequence();
    //cout << "\n>>> set sit " << n << flush;
    return true;
  }
  // set duration
  if (! strncmp(str,"d",1)){
    int n;
    sscanf(str,"%*s %d",&n);
    sequence::duration = n;
    cout << "\n>>> new duration " << sequence::duration << flush;
    return true;
  }
  return false;
}


bool ModTrain::onHearMessage(const char *str) {

  return false;
}


bool ModTrain::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}

void ModTrain::protocol_sit(int terminated){

  if(sequence::loops<2) // forget the first sequence
    return;
  if(protocol_on == false)
    return;
  if(sequence::last_protocolled_at == fld.getTime()  &&
     terminated != GOAL)
    return;

  sequence::last_protocolled_at = fld.getTime();
  if(sequence::loops == parameter::stop_after_loops){ 
    cout << "\nProtokolliere..." << flush;
    fprintf(protf, "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", fld.plType[player_type].player_speed_max, fld.plType[player_type].stamina_inc_max, fld.plType[player_type].player_decay, fld.plType[player_type].inertia_moment, fld.plType[player_type].dash_power_rate, fld.plType[player_type].player_size, fld.plType[player_type].kickable_margin, fld.plType[player_type].kick_rand, fld.plType[player_type].extra_stamina, fld.plType[player_type].effort_max, fld.plType[player_type].effort_min);
    fprintf(protf, "%ld\n", statistics::total_time);
    fflush(protf);
  }

}

void ModTrain::print_statistics(){

}



int ModTrain::check_stop_sequence(){
  Vector player,ball;

  if(fld.getTime() >= sequence::start_time + sequence::duration){
    cout << "\nTIMEOVER" << flush;
    return TIME_OVER;
  }
  if(parameter::stopif_ball_intercepted){
    for (int i = 0 ; i < 11 ; i++){ // do consider goalie
      if(!fld.hisTeam[i].alive) continue;
      player.setX( fld.hisTeam[i].pos_x );
      player.setY( fld.hisTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      if((player-ball).norm() <= parameter::kickrange){
	cout << "\nBALL_INTERCEPTED" << flush;
	return BALL_INTERCEPTED;
      }
    }    
  }


  return CONTINUE;
}


void ModTrain::start_new_sequence(){
  sit.send_cur_situation();  
  sequence::loops ++;
  cout << "\nSequence-Number:" << sequence::loops;
  sequence::start_time = fld.getTime();
}



/* Don't forget this one... */
const char ModTrain::modName[]="ModTrain";

//////////////////////////////////////////////////////////////////////////////////////
//////////////////// Supporting Classes //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////


ostream& operator<< (ostream& o,const Table& t) {
  cout << "[";
  cout.precision(5);
  for (int st= 0; st < t.num_rows; st++) {
    cout << endl << setw(3) << st << ":";
    for (int act= 0; act < t.num_cols; act++) {
      if (act%10==0) cout << endl;
      cout << setw(8) << t(st,act);
    }
  }
  cout << "\n]";
  return o;
}

void Table::set_max_rows(int max) {
  if (max_rows > max)
    return;
  max_rows= max;
  float ** tmp= new float*[max_rows];
  for (int row=0; row<num_rows; row++)
    tmp[row]= tab[row];
  for (int row=num_rows; row < max_rows; row++) 
    tmp[row]= 0;

  delete[] tab;
  tab= tmp;
}

float Table::operator()(int row,int col) const { 
  if (col < 0 || col >= num_cols || row < 0 || row >= num_rows) 
    cerr << "\n(" << row <<"," << col << ") not in range";
  return tab[row][col]; 
}

const float * Table::operator()(int row) const{
  if (row < 0 || row >= num_rows)  
    cerr << "\n(" << row << ") not in range";
  return tab[row];
}

void Table::set(int row, int col, float value) {
  tab[row][col]= value;
}

bool Table::save(const char* fname ) const {
  ofstream out(fname);

  for (int row= 0; row < num_rows; row++) { 
    out << "\n";
    for (int col= 0; col < num_cols; col++) 
      out << " " <<  setw(6) << tab[row][col];
  }

  out.close();
  return true;
}

bool Table::load(int cols,const char* fname ) {
  const int MAX_LINE_LEN= 1024;
  char line[MAX_LINE_LEN];
  
  num_rows= 0;
  num_cols= cols;
  set_max_rows(50);

  FILE *infile=fopen(fname,"r");
  if (infile==NULL){
    fprintf(stderr,"File %s can't be opened\n", fname);
    return false;
  }
  
  float *tmp= 0;

  while(fgets(line,MAX_LINE_LEN,infile)!=NULL){
    if(*line=='\n'||*line=='#')
      continue;  /* skip comments */

    for (int i=0;i <MAX_LINE_LEN; i++) 
      if (line[i]== '\n') {
	line[i]= '\0';
	break;
      }

    if (tmp==0) 
      tmp= new float[num_cols];
    bool warning= false;
    int res= str2val( line, num_cols, tmp);
    if (res == num_cols) {
      if (num_rows >= max_rows)
	set_max_rows(max_rows+50);
      tab[num_rows]= tmp;
      num_rows++;
      tmp= 0;
    }

    if (res != num_cols || warning) {
      cout << "\n problems with reading line = " << line;
      cout << "res= " << res << ", warning = " << warning;
    }
  }
  if (tmp)
    delete[] tmp;
  fclose(infile);  
  return true;
}

string Table::float_to_str(int num){
  std::ostringstream oss;
  for (int i=0; i < num_cols; i++){
    if(i>0) oss << " ";
    oss << tab[num][i];
  }
  return oss.str();
}

/*****************************************************************************/
/*****************************************************************************/

Situations::Situations() {
  num_sit= 0; 
  cur_sit= 0;
  for (int i=0; i<NUM_OBJECTS; i++)
    set_active_obj(i);
}

void Situations::load_table(const char* fname) {
  tab.load(114,fname);
  num_sit= tab.get_num_rows();
}


void Situations::send_situation(int sit_num) {
  if (sit_num<0 || sit_num >= num_sit) return;

  if (!active[0])
    return;
  string buf = tab.float_to_str(sit_num);
  MSG::sendMsg(MSG::MSG_SETSIT, buf);
}
