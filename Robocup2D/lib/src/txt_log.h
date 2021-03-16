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
#ifndef _TXT_LOG_H_
#define _TXT_LOG_H_

#include <iostream>
#include <sstream>
#include <stdio.h>

/******************************************************************************/
/******************************************************************************/

class PosCache {
  struct Pos {
    int time;
    size_t pos;
  };
  static const int MAX_NUM= 4;
  int num;
  int cur;

  Pos tab[MAX_NUM];

public:
  PosCache() { reset(); }
  void reset() { num= 0; cur= 0; }
  void show(std::ostream & out);
  bool put(int time, size_t pos);
  bool get(int time, int & res_time, size_t & res_pos);
};

/******************************************************************************/
/******************************************************************************/

class TextFilter {
public:
  virtual bool process_type_info(int time, const char * dum, char const* & next) {
    std::cout << "\n" << time;
    next= dum;
    return true;
  }

  virtual void process_character(char chr) {
    std::cout << chr;
  }
};

/******************************************************************************/
/******************************************************************************/

class NumberedTextNavigator {
public:
  NumberedTextNavigator();
  ~NumberedTextNavigator();

  bool init_buffer(int size= -1, char * buffer= 0);
  bool open(const char *);
  bool close();

  static int get_recommended_buffer_size() {
    return 2048 + MAX_ADDRESS_SIZE + 2;
  }
  static int get_absolutely_smallest_buffer_size() {
    return 3 * MAX_ADDRESS_SIZE + 2;
  }
  bool show_entries_for_time(int time, TextFilter & ts);
  void test();
protected:
  //static const int MAX_LOG_LINE_SIZE= 29; //29 is just for testing, use larger buffers to be more efficient
  int BUFFER_SIZE;
  int buffer_is_shared;
  int BUFFER_LINE_SIZE; 
  static const int MAX_ADDRESS_SIZE= 16;
  char * line;
  void reset() {
    file= 0;
    pos_cache.reset();
    line= 0;
  }
  FILE * file;
  PosCache pos_cache;

  struct _binary_search_param {
    _binary_search_param() {
      reset();
    }
    void reset() {
      use_start= false;
    }
    //these parameters are set by the corresponding methods
    size_t res_pos;

    //input parameters, use them to influence the behaviour of the search
    bool use_start;
    int start_time;
    size_t start_pos;
    void set_start(int time, size_t pos) {
      use_start= true;
      start_time= time;
      start_pos= pos;
    }
  };

  struct _linear_search_param {
    
    _linear_search_param() {
      reset();
    }
    void reset() {
      use_last_time_and_pos= true;
      use_preferred_time_jump= false;
      use_preferred_time_jump_for_return= false;
      use_num_bytes_limit= false;  
    }
    //these parameters are set by the corresponding methods
    int    first_res_time;
    size_t first_res_pos;
    int    last_res_time;
    size_t last_res_pos;

    bool   first_time_jump_occurred;
    int    first_time_jump_time;
    size_t first_time_jump_pos;

    //input parameters, use them to influence the behaviour of the search
    bool use_last_time_and_pos;
    bool use_preferred_time_jump;
    bool use_preferred_time_jump_for_return;
    bool use_num_bytes_limit;     //useful in binary search (if entries are rare)
    int  preferred_time_jump_time;
    size_t num_bytes_limit;
    void set_preferred_time_jump(int time) {
      use_preferred_time_jump= true;
      use_preferred_time_jump_for_return= true;
      preferred_time_jump_time= time;
    }
    void set_num_bytes_limit( size_t limit ) {
      use_num_bytes_limit= true;
      num_bytes_limit= limit;
    }
  };
  bool forward_linear_search_for_next_time_entry(FILE * file, _linear_search_param & result);
  bool backward_linear_search_for_next_time_entry(FILE * file, _linear_search_param & result);
  bool binary_search_for_time_entry(FILE * file, int targ_time, _binary_search_param & param );
};

#endif
