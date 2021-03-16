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
#include "txt_filter_bs.h"
#include "str2val.h"
#include "macro_msg.h"

void TextFilterCMUlike::reset() { 
  level= -1;
  out= &std::cout;

  show_number= false;
  show_time= false;
  show_level= false;
  show_type= false;
  show_number_in_every_line= false;
  show_time_in_every_line= false;
  show_level_in_every_line= true;
  show_type_in_every_line= false;
    
  for (int i=0; i<TYPE_MAX_NUM; i++) {
    use[i]= false;
  }
  cur_type= TYPE_NORMAL;
  cur_level= 0;
  cur_time= -1;
}

void TextFilterCMUlike::set_normal_mode(int lev) {
  reset();
  level= lev;
  //show_time= true;
  show_level= true;
  show_type= true;
  use[TYPE_NORMAL]= true;
}

void TextFilterCMUlike::set_error_mode(int lev) {
  reset();
  level= lev;
  show_number= true;
  show_level= true;
  show_type= true;
  use[TYPE_ERROR]= true;
}

void TextFilterCMUlike::set_frameview_mode(int lev) {
  reset();
  level= lev;
  use[TYPE_FRAMEVIEW]= true;
}
    
bool TextFilterCMUlike::process_type_info(int time, const char * dum, char const* & next) {
  next= dum;
  cur_time= time;
  if ( next[0] != '.' ) {
    return false;
  }
  next++;
  int tmp;
  if ( ! str2val(next,tmp,next) )
    return false;
  
  strspace(next,next);
  
//  bool got_new_level_information= false;
  
  if ( next[0] == '-' ) {
//    got_new_level_information= true;
    cur_type= TYPE_NORMAL; //can change to TYPE_FRAMEVIEW
    
    if ( ! (use[TYPE_NORMAL] || use[TYPE_FRAMEVIEW]) )
      return false;
    cur_level= 0;
    while ( next[0] == '-' ) {
      cur_level++;
      next++;
    }
    const char * dum2;
    if ( strskip(next,"_2D_",dum2) ) {
      cur_type= TYPE_FRAMEVIEW;
      next= dum2;
    }
  }
  else if ( next[0] == '#' ) {
//    got_new_level_information= true;
    cur_type= TYPE_ERROR;
    if ( ! use[TYPE_ERROR] )
      return false;
    cur_level= 0;
    
    while ( next[0] == '#' ) {
      cur_level++;
      next++;
    }
  }
  
  if ( cur_level > level || !use[cur_type] ) {
    return false;
	}
  

  if ( !use[cur_type] ){
    return false;
	}
  
	process_character('\n');
  //ost << "\n";
  
	/*
  if (show_number) 
    show_cur_number();
    
  if (show_time) {
    ost << time << ".0 ";
  }
	*/
  
  /*
  if (got_new_level_information && show_level 
      || show_level_in_every_line)
    show_cur_level();
    */
  
  return true;
}

/******************************************************************************/
/******************************************************************************/

void CmdCount::reset() {
  move_count= 0;
  kick_count= 0;
  dash_count= 0;
  turn_count= 0;
  catch_count= 0;
}

void CmdCount::show(std::ostream & out) const {
  out << "(m= " << move_count
      << ", k= " << kick_count
      << ", d= " << dash_count
      << ", t= " << turn_count
      << ", c= " << catch_count
      << ")";
}

void CmdCount::show_greater_counts(std::ostream & out, const CmdCount & c) const {
  if (move_count > c.move_count)
    out << " move";
  if (kick_count > c.kick_count)
    out << " kick";
  if (dash_count > c.dash_count)
    out << " dash";
  if (turn_count > c.turn_count)
    out << " turn";
  if (catch_count > c.catch_count)
    out << " catch";
}

void CmdCount::set_neg_counts_to_zero() {
  if (move_count<0) move_count= 0;
  if (kick_count<0) kick_count= 0;
  if (dash_count<0) dash_count= 0;
  if (turn_count<0) turn_count= 0;
  if (catch_count<0) catch_count= 0;
}

void CmdCount::operator +=(const CmdCount & count) {
  move_count += count.move_count; 
  kick_count += count.kick_count; 
  dash_count += count.dash_count; 
  turn_count += count.turn_count; 
  catch_count+= count.catch_count; 
}

void CmdCount::operator -=(const CmdCount & count) {
  move_count -= count.move_count; 
  kick_count -= count.kick_count; 
  dash_count -= count.dash_count; 
  turn_count -= count.turn_count; 
  catch_count-= count.catch_count; 
}

bool CmdCount::operator >=(const CmdCount & count) const {
  return 
    move_count >= count.move_count && 
    kick_count >= count.kick_count && 
    dash_count >= count.dash_count &&
    turn_count >= count.turn_count &&
    catch_count>= count.catch_count; 
}

std::ostream & operator<< (std::ostream& o, const CmdCount & v) {
  v.show(o);
  return o;
}  

CmdCount operator-(const CmdCount & c1, const CmdCount & c2) {
  CmdCount ret;
  ret.move_count=  c1.move_count -  c2.move_count; 
  ret.kick_count=  c1.kick_count -  c2.kick_count; 
  ret.dash_count=  c1.dash_count -  c2.dash_count; 
  ret.turn_count=  c1.turn_count -  c2.turn_count; 
  ret.catch_count= c1.catch_count - c2.catch_count; 
  return ret;
}

bool operator!=(const CmdCount & c1, const CmdCount & c2) {
  return 
    c1.move_count != c2.move_count ||
    c1.kick_count != c2.kick_count ||
    c1.dash_count != c2.dash_count ||
    c1.turn_count != c2.turn_count ||
    c1.catch_count != c2.catch_count;
}
