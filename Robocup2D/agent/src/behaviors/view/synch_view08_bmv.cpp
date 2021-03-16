#include "synch_view08_bmv.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "blackboard.h"
                                                             
const int BASELEVEL=3;		// define for log macros

bool SynchView08::initialized=false;

SynchView08::SynchView08() {
  init_state = 1; // assume synching works nicely
  cyc_prob_synched = 0;
  goalie_last_normal = 0;
  missed_commands  = 0;
  cyc_cnt = 0;
  asynch_view = new BS03View();
}

bool SynchView08::get_cmd(Cmd & cmd){
  if(WSinfo::ws->ms_time_of_see < 0 ) 
  { // ignore_see, so no view behavior...
    LOG_POL(BASELEVEL+1,<<"SynchView08: See messages are being ignored, thus we don't use a view behavior.");
    change_view(cmd,WSinfo::ws->view_angle,WSinfo::ws->view_quality); // set Blackboard
    return false;
  }
  if(!(WSinfo::ws->synch_mode_ok || Blackboard::get_guess_synched()))
  {
  	init_state = 0; // change to check mode in order to check why synch_mode is not set
  }
  cyc_cnt++;
  LOG_POL(BASELEVEL+2,<<"SynchView0808: Time of see is "<<time_of_see());
  LOG_POL(BASELEVEL+3,<<"SynchView08: ms_sb: "<<WSinfo::ws->ms_time_of_sb
	  <<", ms_see: "<<WSinfo::ws->ms_time_of_see<<", cyc_cnt: "<<cyc_cnt);
  if(WSinfo::ws->time % 500 == 5 || WSinfo::ws->time == 5999) {
    if(missed_commands>0) {
      ERROR_OUT << "\nSynchView08 [p="<<WSinfo::me->number<<" t="<<WSinfo::ws->time
		<<"]: Missed "<<missed_commands<<" change_view within the last 500 cycles!";
    }
  }
  
  if(WSinfo::ws->ms_time_of_see<WSinfo::ws->ms_time_of_sb) {
    //ERROR_OUT << "\nMissed a see command!";
    LOG_POL(BASELEVEL,<<"SynchView08: Missed a see command!");
  }
  switch(init_state) {
  case 0:  // something went wrong so we better check if synch_see works correctly
  	if(WSinfo::ws->synch_mode_ok || Blackboard::get_guess_synched())
  	{   /* 
  		   if (ok synch_mode) was received we cannot switch back to asynch mode!
  		   thus the problem that occured must be due to some environmental poblems
  		   and we keep our std synched view strategy
  		 */
  		LOG_POL(BASELEVEL,<<"SynchView08: Detected connection problems!");
  		init_state = 1;
  		change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
    	return true;
  	}
    if ( time_of_see() > ServerOptions::synch_see_offset - 3 && time_of_see() < ServerOptions::synch_see_offset + 3 ) 
    {   // if time_of_see equals 30 we might be in synch mode but we  
    	// just did not catch the (ok synch see) message
      if(cyc_prob_synched == cyc_cnt - 1)
      { /* 
      	   there arrived 2 see messages at 30 ms therefore it is very likely
      	   that we simply missed (ok synch see) -> switching back to synched mode
      	   but keep on checking for correct message arrival
      	 */
      	init_state = 1;
      	Blackboard::set_guess_synched();
      	cyc_prob_synched = 0;      
      }
      cyc_prob_synched = cyc_cnt;
      // not sure yet ... try to get another synch see message
      change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    } 
    // synch mode does not work so we better use asynchronous viewing
    LOG_ERR(0,<< "ERROR: synch see mode not set, switching to asynchronous see mode!");
  	return asynch_view->get_cmd(cmd);
  	break;
  case 1:  // normal mode we use synch viewing 
	if(    ClientOptions::consider_goalie 
        && (WSinfo::me->pos-WSinfo::ball->pos).norm() > 30
      ) 
    {
      LOG_POL(BASELEVEL+1,<<"SynchView08 in goalie mode: Ball far away, using normal width");
      goalie_last_normal = cyc_cnt;
      change_view(cmd,Cmd_View::VIEW_ANGLE_NORMAL,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    }
    if ((time_of_see() >= (40*ServerOptions::slow_down_factor) && goalie_last_normal != cyc_cnt -1)
         || (goalie_last_normal == cyc_cnt -1  && time_of_see() >= (140*ServerOptions::slow_down_factor))
       )
    {
		// Message arrived at least 10 ms off, there must be something wrong here
		init_state=0;
		LOG_POL(BASELEVEL+0,<<"SynchView08: See message too late -> out of sync!");
		change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
		return true;
    }
    LOG_POL(BASELEVEL+0,<<"SynchView08: Synchronous viewing ok ");
    change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
    return true;
    break;    
  }
  return false;
}

bool SynchView08::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized;
  initialized = BS03View::init(conf_file, argc, argv);
  if(!initialized) { std::cout << "\nSynchView08 behavior NOT initialized!!!"; }
  return initialized;
}

void SynchView08::change_view(Cmd &cmd,int width, int quality) {

  if(WSinfo::is_ball_pos_valid() && WSinfo::ball->age<=1) {
    Blackboard::set_last_ball_pos(WSinfo::ball->pos);
  }
  if(WSinfo::ws->ms_time_of_see >= 0) {  // ignore_see, thus nothing happens here
    if(WSinfo::ws->time>0 && WSinfo::ws->view_angle!=next_view_width) {
      LOG_ERR(0,<<"SynchView08: WARNING: Server missed a change_view!");
      missed_commands++;
    }
  }
  next_view_width = width;
  Blackboard::set_next_view_angle_and_quality(width,quality);
  LOG_POL(BASELEVEL+0,<<"SynchView08: Setting width "<<width<<", quality "<<quality);
  if(WSinfo::ws->view_angle!=width || WSinfo::ws->view_quality!=quality) {
    cmd.cmd_view.set_angle_and_quality(width,quality);
  }
}

/**
 * The method time_of_see() returns a value in MILLISECONDS that tells
 * about the difference between the arrival time of the last see message
 * and sense-body message received.
 * 
 * Since this behaviour uses the synchronized view model the see message
 * should always arrive after sense of body!
 */
int SynchView08::time_of_see() {
  int tos;
  long sb = WSinfo::ws->ms_time_of_sb;
  long see = WSinfo::ws->ms_time_of_see;
  if(see>=sb) 
    tos = see-sb;
  else 
    tos = see - sb + (int)get_delay();
  return tos;
}

double SynchView08::get_delay() {
  double view_delay=0;
  switch(WSinfo::ws->view_angle) 
  {
    case Cmd_View::VIEW_ANGLE_NARROW:
      view_delay = 100*ServerOptions::slow_down_factor;
      break;
    case Cmd_View::VIEW_ANGLE_NORMAL:
      view_delay = 200*ServerOptions::slow_down_factor;
      break;
    case Cmd_View::VIEW_ANGLE_WIDE:
      view_delay = 300*ServerOptions::slow_down_factor;
      break;
  }
  return view_delay;
}
