/*
 *  Author:   Artur Merke 
 */
#ifndef _MESSAGES_TIMES_H_
#define _MESSAGES_TIMES_H_

#include <iostream>
#include "basics/wmdef.h"

struct MessagesTimes {
  static void init(bool create_buffer);

  //static const int MAX_SIZE= 30000;
  static int MAX_SIZE;

  struct Entry {
    short ms_time_diff;
    short time;
    short type;
    short param;
  };

  static long  last_msg_ms_time[MESSAGE_MAX];
  static short last_msg_time[MESSAGE_MAX];
  static long ms_time_last;
  static int size;
  static Entry * entry;

  static void reset();

  static void warning(std::ostream & out, const short * msg_time, const long * msg_ms_time, int msg_idx, long ms_time);
  static void add(int time, long ms_time, int type, int param= 0);

  static void add_before_behave(int time, long ms_time) {
    add(time,ms_time,MESSAGE_BEFORE_BEHAVE);
  }

  static void add_after_behave(int time, long ms_time) {
    add(time,ms_time,MESSAGE_AFTER_BEHAVE);
  }

  static void add_lost_cmd(int time, int cmd_type) {
    add(time,ms_time_last,MESSAGE_CMD_LOST, cmd_type);
  }

  static void add_sent_cmd(int time, int cmd_type) {
    add(time,ms_time_last,MESSAGE_CMD_SENT, cmd_type);
  }

  static bool save(std::ostream & out);

  static bool save(const char * fname);
};

#endif
