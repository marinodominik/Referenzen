#include "bs03_view_bmv.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "blackboard.h"

#define VIEW_NORMAL_TRESHOLD (14*ServerOptions::slow_down_factor)
                                 // see message must arrive earlier than this to enable
                                 // normal view angle!
                                 // the server's slow down factor is also taken
                                 // into consideration

#define BASELEVEL 3

bool BS03View::initialized = false;

BS03View::BS03View() {
  init_state=0;
  last_normal=-1;
  lowq_cycles = lowq_counter = 0;
  missed_commands  = 0;
  cyc_cnt = 0;
  //missed_sees = 0;
  cur_view_width = next_view_width = -1;
  need_synch = false;
}

BS03View::~BS03View() {

}

bool BS03View::get_cmd(Cmd & cmd){
  if(WSinfo::ws->ms_time_of_see < 0 ) { // ignore_see, so no view behavior...
    LOG_POL(BASELEVEL+1,<<"BS03View: See messages are being ignored, thus we don't use a view behavior.");
    change_view(cmd,WSinfo::ws->view_angle,WSinfo::ws->view_quality); // set Blackboard
    return false;
  }
  cyc_cnt++;
  LOG_POL(BASELEVEL+2,<<"BS03View: Time of see is "<<time_of_see());
  LOG_POL(BASELEVEL+3,<<"BS03View: ms_sb: "<<WSinfo::ws->ms_time_of_sb
	  <<", ms_see: "<<WSinfo::ws->ms_time_of_see<<", cyc_cnt: "<<cyc_cnt);
  if(WSinfo::ws->time % 500 == 5 || WSinfo::ws->time == 5999) {
    if(missed_commands>0) {
      ERROR_OUT << "\nBS03View [p="<<WSinfo::me->number<<" t="<<WSinfo::ws->time
		<<"]: Missed "<<missed_commands<<" change_view within the last 500 cycles!";
    }
  }
  if(WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW) lowq_cycles++;
  else lowq_cycles=0;
  if(WSinfo::ws->play_mode==PM_PlayOn) Blackboard::force_highq_view=false;
  
  if(WSinfo::ws->ms_time_of_see<WSinfo::ws->ms_time_of_sb) {
    //ERROR_OUT << "\nMissed a see command!";
    LOG_POL(BASELEVEL,<<"BS03View: Missed a see command!");
  }

  switch(init_state) {
  case 0:  // need to start synching! 
    LOG_POL(BASELEVEL+0,<<"BS03View: Still initialising synching sequence.");
    if ( time_of_see() > (10*ServerOptions::slow_down_factor) ) 
      init_state=1;
    change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_LOW);
    return true;
    break;
  case 1:
    LOG_POL(BASELEVEL+0,<<"BS03View: Synching to sense_body in progress...");
    if(can_view_normal_strict()) {  // we are synched now, so start normal behaviour!
      init_state=2;
      LOG_POL(BASELEVEL+0,<<"BS03View: Synching complete, starting normal behaviour!");
      last_normal=WSinfo::ws->time;
      change_view(cmd,Cmd_View::VIEW_ANGLE_NORMAL,Cmd_View::VIEW_QUALITY_HIGH);
      cyc_cnt=0;
      return true;
    }
    if(WSinfo::ws->play_mode==PM_PlayOn) {
      LOG_POL(BASELEVEL+0,<<"BS03View: play_on, stop synching!");
      change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    }
    if(Blackboard::force_highq_view) {
      LOG_POL(BASELEVEL+0,<<"BS03View: force_highq_view set, stop synching!");
      change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    }
    if(WSinfo::ws->play_mode==PM_my_BeforePenaltyKick
       || WSinfo::ws->play_mode==PM_his_BeforePenaltyKick
       || WSinfo::ws->play_mode==PM_my_PenaltyKick
       || WSinfo::ws->play_mode==PM_his_PenaltyKick) {
      LOG_POL(BASELEVEL+0,<<"BS03View: penalty mode, no synching...");
      change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    }
    change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_LOW);
    return true;
    break;    
  case 2:
    if(    ClientOptions::consider_goalie 
        && (WSinfo::me->pos-WSinfo::ball->pos).norm() > 30
        && (   time_of_see()==0 
            || time_of_see()==(50*ServerOptions::slow_down_factor))) 
    {
      LOG_POL(BASELEVEL+1,<<"BS03View in goalie mode: Ball far away, using normal width");
      last_normal=WSinfo::ws->time;
      change_view(cmd,Cmd_View::VIEW_ANGLE_NORMAL,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    } 
    else 
    {
       if (time_of_see() >= (68*ServerOptions::slow_down_factor) 
          /*&& WSinfo::view_angle!=Cmd_View::VIEW_ANGLE_WIDE*/) 
       {
	// Message should never arrive that late - out of sync!
		init_state=1;
		LOG_POL(BASELEVEL+0,<<"BS03View: See message too late -> out of sync!");
		change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
		return true;
       }

       if(can_view_normal()) 
       {  
         LOG_POL(BASELEVEL+1,<<"BS03View: Switching to normal (t="<<WSinfo::ws->time<<")");
         last_normal=WSinfo::ws->time;
         change_view(cmd,Cmd_View::VIEW_ANGLE_NORMAL,Cmd_View::VIEW_QUALITY_HIGH);
         return true;
       } 
       else 
       {
	//if(mdpInfo::play_mode()==MDPstate::PM_PlayOn && !mdpInfo::am_I_goalie() &&
	//   (WSinfo::ws->time_of_last_update!=WSinfo::ws->time || ms_time_since_sb()>70)) {
	//  LOG_POL(2,<<"BS03_View: WARNING: See message apparently arrived later than 70 ms!");
	  //cerr <<"BS03_View: WARNING: See message apparently arrived later than 70 ms!";
	//}

	//init_state=1;
	//return change_view(Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_LOW);
	//} else {
		LOG_POL(BASELEVEL+1,<<"BS03View: Switching to narrow.");
		if(   WSinfo::ws->time-last_normal>20 
		   && WSinfo::ws->play_mode!=PM_PlayOn
		   && !Blackboard::force_highq_view
		   && ! (    WSinfo::ws->play_mode==PM_my_BeforePenaltyKick
    			  || WSinfo::ws->play_mode==PM_his_BeforePenaltyKick
				  || WSinfo::ws->play_mode==PM_my_PenaltyKick
 				  || WSinfo::ws->play_mode==PM_his_PenaltyKick) )   // 10
	    {
		  LOG_POL(BASELEVEL+0,<<"BS03View: Could not switch to normal view since more than 20 cycles!");
		  LOG_POL(BASELEVEL+0,<<"BS03View: Out of sync? Synching again (not in play_on mode).");
		  //cerr << "\n\n#"<<WSinfo::me->number<<" PROBLEM mit BS03_View";
		  init_state=1;
		  change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_LOW);
		  return true;
		}
    	change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
    	return true;
	//}
      }
    }
    break;
  case 4:  // new mode: do not check every cycle!  * DOES NOT WORK *
    if ( cyc_cnt % 3==0 ) {
      change_view(cmd,Cmd_View::VIEW_ANGLE_NORMAL,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    } 
    else 
    {
      change_view(cmd,Cmd_View::VIEW_ANGLE_NARROW,Cmd_View::VIEW_QUALITY_HIGH);
      return true;
    }
    break;
  }
  return false;
}

bool BS03View::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized;
  initialized = true;
  if(!initialized) { std::cout << "\nBS03View behavior NOT initialized!!!"; }
  return initialized;
}

void BS03View::change_view(Cmd &cmd,int width, int quality) {
  if(quality==Cmd_View::VIEW_QUALITY_LOW) {
    Blackboard::set_last_lowq_cycle(WSinfo::ws->time);
  } 
  //if(WSinfo::ws->view_quality==Cmd_View::VIEW_QUALITY_HIGH) {
  //  Blackboard::set_last_ball_pos(WSinfo::ball->pos);
  //}
  if(WSinfo::is_ball_pos_valid() && WSinfo::ball->age<=1) {
    Blackboard::set_last_ball_pos(WSinfo::ball->pos);
  }
  if(WSinfo::ws->ms_time_of_see >= 0) {  // ignore_see, thus nothing happens here
    if(WSinfo::ws->time>0 && WSinfo::ws->view_angle!=next_view_width) {
      LOG_ERR(0,<<"BS03View: WARNING: Server missed a change_view!");
      missed_commands++;
      //ERROR_OUT << "\n### Player "<<WSinfo::me->number<<", Cycle "<<WSinfo::ws->time
      //		<<": Probably missed a change_view command!";
    }
#if 0
    // this check is not needed and counter productive in chaotic environments... 
    if(width==Cmd_View::VIEW_ANGLE_NORMAL && !ClientOptions::consider_goalie 
       && WSinfo::ws->view_angle==Cmd_View::VIEW_ANGLE_NORMAL) {   // niemals zweimal NORMAL!
      ERROR_OUT << "\n### Player "<<WSinfo::me->number<<", Cycle "<<WSinfo::ws->time
		<< ": Why should I view NORMAL again? Ignoring!";
      width=Cmd_View::VIEW_ANGLE_NARROW;
    }
#endif
    // now check if we are synching too long due to network problems...
    if(quality == Cmd_View::VIEW_QUALITY_LOW) {
      if(lowq_counter-->0) {
	LOG_POL(BASELEVEL+0,<<"BS03View: Still waiting some cycles before switching to low again!");
	quality = Cmd_View::VIEW_QUALITY_HIGH;
      }
      else if(lowq_cycles>10) {
	LOG_ERR(0,<<"BS03View: WARNING: More than 10 cycles in low quality, ignoring!");
	ERROR_OUT << "\n###Player "<<WSinfo::me->number<<", Cycle "<<WSinfo::ws->time
		  << ": Could not sync (network probs?), switching to HIGH!";
	lowq_counter = 8;  // cycles to wait before next LOW
	quality = Cmd_View::VIEW_QUALITY_HIGH;
      }
    } else {
      lowq_counter = 0;
    }
  }
  next_view_width = width;
  Blackboard::set_next_view_angle_and_quality(width,quality);
  LOG_POL(BASELEVEL+0,<<"BS03View: Setting width "<<width<<", quality "<<quality);
  if(WSinfo::ws->view_angle!=width || WSinfo::ws->view_quality!=quality) {
    cmd.cmd_view.set_angle_and_quality(width,quality);
  }
}

/**
 * The method time_of_see() returns a value in MILLISECONDS that tells
 * about the difference between the arrival time of the last see message
 * and sense-body message received.
 * 
 * The normal case is, that the see message arrived AFTER the sense-body
 * message. In this case, simply the difference between both arrival times
 * is returned.
 * 
 * In case that the last message received was a sense-body message, the
 * difference see-sb will be negative. Here, a constant c is added to that
 * negative value (so that in turn the value returned will be positive).
 * The constant c is determined via function get_delay() which tells about
 * the actual frequency of see messages arriving from the server.
 */
int BS03View::time_of_see() {
  int tos;
  long sb = WSinfo::ws->ms_time_of_sb;
  long see = WSinfo::ws->ms_time_of_see;
  //if(WSinfo::ws->time==WSinfo::ws->time_of_last_update) {
  if(see>=sb) 
    tos = see-sb;
  else 
    tos = see - sb + (int)get_delay();
  //return ((tos+5)/10)*10;
  return tos;
}

double BS03View::get_delay() {
  double view_delay=0;
  switch(WSinfo::ws->view_angle) 
  {
    case Cmd_View::VIEW_ANGLE_NARROW:
      view_delay = 75*ServerOptions::slow_down_factor;
      break;
    case Cmd_View::VIEW_ANGLE_NORMAL:
      view_delay = 150*ServerOptions::slow_down_factor;
      break;
    case Cmd_View::VIEW_ANGLE_WIDE:
      view_delay = 300*ServerOptions::slow_down_factor;
      break;
  }
  switch(WSinfo::ws->view_quality) 
  {
    case Cmd_View::VIEW_QUALITY_LOW:
      view_delay/=2.0;
  }
  return view_delay;
}

bool BS03View::can_view_normal() {
  if(time_of_see()<VIEW_NORMAL_TRESHOLD) return true;
  else return false;
}

bool BS03View::can_view_normal_strict() {
  if (time_of_see() < (8*ServerOptions::slow_down_factor)) 
    return true;
  else return false;
}
