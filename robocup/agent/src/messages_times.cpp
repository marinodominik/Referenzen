/*
 *  Author:   Artur Merke 
 */

#include "messages_times.h"
#include <iomanip>
#include <fstream>

#include "basics/WM.h"
#include "macro_msg.h"
#include "server_options.h"

long  MessagesTimes::last_msg_ms_time[MESSAGE_MAX];
short MessagesTimes::last_msg_time[MESSAGE_MAX];
long  MessagesTimes::ms_time_last= 0;
int   MessagesTimes::size= 0;
int   MessagesTimes::MAX_SIZE= 0;

MessagesTimes::Entry * MessagesTimes::entry= 0;

void MessagesTimes::init(bool create_buffer) {
  reset();
  if ( create_buffer ) {
    MAX_SIZE= 30000;
    entry= new Entry[MAX_SIZE];
  }
}

void MessagesTimes::reset() {
  size= 0;
  MAX_SIZE= 0;
  for (int i=0; i<MESSAGE_MAX; i++) {
    last_msg_time[i]= 0;
    last_msg_ms_time[i]= 0;
  }
}

void MessagesTimes::warning(std::ostream & out, const short * msg_time, const long * msg_ms_time, int msg_idx, long ms_time) {
  const char * PREFIX= "@@ ";
  
  Entry & e= entry[msg_idx];
  
  switch ( e.type ) {
  case MESSAGE_SENSE_BODY:
    if ( e.time >  msg_time[MESSAGE_SENSE_BODY]+1 ) {
      out << PREFIX << "error: one/more SENSE_BODY was skipped";
      return;
    }
    if ( e.time > msg_time[MESSAGE_BEFORE_BEHAVE]+1 ) {
      out << PREFIX << "error: no behave in the time(s) before";
      return;
    }
    if ( ms_time - msg_ms_time[MESSAGE_SENSE_BODY] > (110*ServerOptions::slow_down_factor)
	 || ms_time - msg_ms_time[MESSAGE_SENSE_BODY] < (90*ServerOptions::slow_down_factor) ) {
      out << PREFIX << "warning: time diff from last SENSE_BODY  < (90*ServerOptions::slow_down_factor) or > (110*ServerOptions::slow_down_factor) ms: " << ms_time - msg_ms_time[MESSAGE_SENSE_BODY];
      return;
    }
    break;
  case MESSAGE_BEFORE_BEHAVE: 
    if ( e.time != msg_time[MESSAGE_SEE] ) {
      out <<  "info: no SEE before behave";
      return;
    }
    break;
  case MESSAGE_AFTER_BEHAVE: 
    if ( e.time != msg_time[MESSAGE_SENSE_BODY] ) {
      out << PREFIX << "error: no SENSE_BODY with this time";
      return;
    }
    
    if ( ms_time - msg_ms_time[MESSAGE_SENSE_BODY] > (110*ServerOptions::slow_down_factor) ) {
      out << PREFIX << "error: " << ms_time - msg_ms_time[MESSAGE_SENSE_BODY] << " ms after SENSE_BODY";
      return;
    }
    
    if ( ms_time - msg_ms_time[MESSAGE_SENSE_BODY] > (90*ServerOptions::slow_down_factor) ) {
      out << PREFIX << "warning: " << ms_time - msg_ms_time[MESSAGE_SENSE_BODY] << " ms after SENSE_BODY";
      return;
    }
    break;
  case MESSAGE_CMD_LOST:
    out << PREFIX << "error: " << CMD_STRINGS[ e.param ];
    break;
  }
}

void MessagesTimes::add(int time, long ms_time, int type, int param) {
  if (time <= 0 || time==3000) //don't store all the info in time 0
    return;

  if (size < MAX_SIZE) {
    entry[size].time= time;
    entry[size].type= type;
    entry[size].param= param;
    entry[size].ms_time_diff= short(ms_time- ms_time_last);
    
    ms_time_last= ms_time;
    
    last_msg_ms_time[type]= ms_time;
    last_msg_time[type]= time;
    size++;
  }
}

bool MessagesTimes::save(std::ostream & out) {
  long ms_time= 0;

  long ms_times[MESSAGE_MAX];
  short times[MESSAGE_MAX];

  for (int i=0; i<MESSAGE_MAX; i++) {
    ms_times[i]= 0;
    times[i]= 0;
  }

  for (int i= 0; i < size; i++) {
    ms_time+= entry[i].ms_time_diff;
    int type= entry[i].type;

    if (type == MESSAGE_CMD_LOST ) {
      out << "\n" << entry[i].time << ".0 # lost command " << CMD_STRINGS[ entry[i].param ];
      continue;
    }
    
    if (type == MESSAGE_SENSE_BODY) 
      out << "\n" << entry[i].time << ".0 - cycle | ms_time | diff to last msg | diff to last msg of same type | diff to last sense body msg";
    
    out << "\n" << entry[i].time << ".0 - " 
	<< std::setw(8) << entry[i].time 
	<< " " << std::setw(8) << ms_time
	<< " " << std::setw(8) << entry[i].ms_time_diff;
    if (i>0) {
      out << std::setw(8) << ms_time - ms_times[type]
	  << std::setw(8) << ms_time - ms_times[MESSAGE_SENSE_BODY];
    }
    else
      out << std::setw(8) << 0 << std::setw(8) << 0;

    if ( type >= 0 && type < MESSAGE_MAX) 
      out << " " << MESSAGE_STRINGS[type];
    else
      out << " ?????????????";
    //out << " " << type;

    if ( type == MESSAGE_CMD_LOST || type == MESSAGE_CMD_SENT )
      out << " cmd= " << CMD_STRINGS[ entry[i].param ];
    else if ( type == MESSAGE_DUMMY1 || type == MESSAGE_DUMMY2)
      out << " par= " << entry[i].param;

    //this code was added, to be machine readable by rcss_comm_check
    if ( type == MESSAGE_CMD_SENT )
      out << "\n" << entry[i].time << ".0 -- sent_cmd " << CMD_STRINGS[ entry[i].param ];
    else if ( type == MESSAGE_CMD_LOST )
      out << "\n" << entry[i].time << ".0 -- lost_cmd " << CMD_STRINGS[ entry[i].param ];

    warning(out, times, ms_times,i, ms_time);

    ms_times[type]= ms_time;
    times[type]= entry[i].time;
  }
  if (size == MAX_SIZE) 
    out << "\n(full)";
  return true;
}

bool MessagesTimes::save(const char * fname) {
  std::ofstream out(fname);
  if (!out) {
    ERROR_OUT << ID << "\ncannot open file " << fname;
    return false;
  }
  bool res= save(out);
  out.close();
  return res;
}
