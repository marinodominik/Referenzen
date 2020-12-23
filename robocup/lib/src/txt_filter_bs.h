/*
 * Copyright (c) 2002 - , Artur Merke <amerke@ira.uka.de> 
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */
#ifndef _TXT_FILTER_BS_H_
#define _TXT_FILTER_BS_H_

#include <string.h>
#include "txt_log.h"
#include "str2val.h"
#include "macro_msg.h"
#include "stl_string_regex.h"

struct TextFilterCMUlike : public TextFilter {
  void show_cur_number() {
    *out << "[P=" << cur_number << "] ";
  }

  void show_cur_level(int l=-10, int t=-10) {
		if(l==-10){
			l = cur_level;
			t = cur_type;
		}
    for (int i= 0; i < l; i++) {
      if (t == TYPE_NORMAL)
	*out << "-";
      else if ( t == TYPE_ERROR )
	*out << "#";
    }
  }

  int cur_number;
  int cur_time;
  int cur_type;
  int cur_level;

  std::ostringstream ost;

  static const int TYPE_NORMAL= 0;
  static const int TYPE_ERROR= 1;
  static const int TYPE_FRAMEVIEW= 2;
  static const int TYPE_MAX_NUM= 3;

  bool use[TYPE_MAX_NUM];

  int level;
  bool show_number;
  bool show_number_in_every_line;
  bool show_time;
  bool show_time_in_every_line;
  bool show_level;
  bool show_level_in_every_line;
  bool show_type;
  bool show_type_in_every_line;

  char* logsearchpat;

  std::ostream * out;
public:
  TextFilterCMUlike() { cur_number= -1; reset(); } 

  void reset();
  void set_stream(std::ostream & o) {
    out= &o;
  }
  void set_number(int number) {
    cur_number= number;
  }

  inline void set_logsearchpat(char*s){logsearchpat=s;};
  void set_normal_mode(int lev);
  void set_error_mode(int lev);
  void set_frameview_mode(int lev);
  bool process_type_info(int time, const char * dum, char const* & next);
  void process_character(char chr) {
		static int localType  = cur_type;
		static int localLevel = cur_level;

		if(chr!='\n'){
			localType  = cur_type;
			localLevel = cur_level;
		}
		
    if ( (localLevel > level || !use[localType]))
				return;

		if(chr == '\n'){
			ost.flush();
			if( ost.str().length()>0 &&
					(localType!=TYPE_NORMAL || !logsearchpat || !strlen(logsearchpat) || regx_match(ost.str(),logsearchpat))){

				if (localType == TYPE_NORMAL && show_number_in_every_line)
					show_cur_number();
				if (localType == TYPE_NORMAL && show_time_in_every_line)
					*out << cur_time << ".0 ";

				if (localType == TYPE_NORMAL && show_level_in_every_line)
					show_cur_level(localLevel,localType);

        //TG08: changed to reestablish corrupted cout output stream
        if (    (*out).good() == false
             && out == &(std::cout) )
        { 
          printf("cout is broken, error state is %d \n", out->rdstate()); 
          out->clear(); 
        }

				*out << ost.str()<<std::endl;
				(*out).flush();
				ost.str("");
				ost.flush();
			}else{
				ost.str("");
				ost.flush();
				return;
			}
		}
    else{
      ost<<chr;
      return;
    }

    // *out << chr;

    return;
    if (chr != '\n') 
      return;

    if (show_number_in_every_line)
      show_cur_number();

    if (show_time_in_every_line)
      ost << cur_time << ".0 ";

    if (show_level_in_every_line) 
      show_cur_level();
  }
};

/******************************************************************************/
/******************************************************************************/

struct CmdCount {
  int move_count;
  int kick_count;
  int dash_count;
  int turn_count;
  int catch_count;
  CmdCount() { reset(); }
  void reset();
  int total_count() const { return move_count + kick_count + dash_count + turn_count + catch_count; }
  void show_greater_counts(std::ostream & out, const CmdCount & c) const;
  void show(std::ostream & out) const;

  void set_neg_counts_to_zero();

  void operator +=(const CmdCount & count);
  void operator -=(const CmdCount & count);
  bool operator >=(const CmdCount & count) const;
};

std::ostream & operator<< (std::ostream& o, const CmdCount & v);  
CmdCount operator-(const CmdCount & c1, const CmdCount & c2);
bool operator!=(const CmdCount & c1, const CmdCount & c2);

/******************************************************************************/
/******************************************************************************/

/*struct TextFilterCmdCounter : TextFilter {
  CmdCount count;
  bool got_such_time;
  TextFilterCmdCounter() { reset(); }
  void reset() { 
    got_such_time= false;
    count.reset();
  }
  bool process_type_info(int time, const char * dum, char const* & next) {
    //cout << "\n" << time;
    got_such_time= true;

    if ( ! strskip(dum,".0 -",next) ) 
      return true;
    
    dum= next;
    //cout << "\n######## dum1= "; for (int i=0; i<10 && dum[i] != '\0'; i++) cout << dum[i];

    while (*dum == '-')
      dum++;
      
    if ( ! strskip(dum,"sent_cmd ",next) ) 
      return true;

    //cout << "\n######## dumE= "; for (int i=0; i<20 && dum[i] != '\0'; i++) cout << dum[i];

    dum= next;
    if ( strskip(dum,"dash",next) ) {
      count.dash_count++;
      return true;
    }
    if ( strskip(dum,"turn",next) ) {
      count.turn_count++;
      return true;
    }
    if ( strskip(dum,"kick",next) ) {
      count.kick_count++;
      return true;
    }
    if ( strskip(dum,"catch",next) ) {
      count.catch_count++;
      return true;
    }
    if ( strskip(dum,"moveto",next) ) {
      count.move_count++;
      return true;
    }
    
    next= dum;
    return true;
  } 

  void process_character(char chr) {} //do nothing
};*/



#endif
