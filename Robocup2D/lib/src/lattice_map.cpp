/*
 * Copyright (c) 2000 - 2001, Artur Merke <amerke@ira.uka.de> 
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
#include "lattice_map.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "str2val.h"

using namespace std;

#include "macro_msg.h"

#ifndef ERROR_STREAM
#define ERROR_STREAM std::cerr
#endif

#ifndef ERROR_OUT 
#define ERROR_OUT ERROR_STREAM << "\n\n*** ERROR file=\"" << __FILE__ << "\" line=" <<__LINE__
#endif

#ifndef WARNING_STREAM
#define WARNING_STREAM std::cerr
#endif

#ifndef WARNING_OUT
#define WARNING_OUT WARNING_STREAM << "\n\n*** WARNING file=\"" << __FILE__ << "\" line=" << __LINE__
#endif


const char * const LatticeOpt::DIM_STR= "DIM";
const char * const LatticeOpt::KACZMARZ_STR= "KACZMARZ";

/*****************************************************************************/


void LatticeSlice::append_array(int len, const float * arr) {
  if ( len <= 0 )
    return;

  int new_len= num + len;

  if (new_len <= 0)
    return;

  float *new_arr= new float[new_len];
  
  if (num > 0)
    memcpy( (void*)new_arr,(void*)support, sizeof(float) * num);

  if (len > 0)
    memcpy( (void*)(new_arr+num),(void*)arr, sizeof(float) * len);

  if (support)
    delete[] support;

  num= new_len;
  support= new_arr;
}

bool LatticeSlice::read(const char * str) {
  num= 0;
  if ( support ) {
    delete[] support;
    support= 0;
  }
  const char * dum= str;
  const int arr_size_max= 100; //must be > 3;
  float arr[arr_size_max];
  int arr_size= 0;

  while ( dum ) {
    const char * dum2;
    int tmp= str2val(dum, arr[arr_size], dum2);
    if ( tmp == 1 ) {
      //ERROR_OUT << "\narr[" << arr_size << "]= " << arr[arr_size];
      arr_size+= 1;
      if ( arr_size > arr_size_max ) {
	append_array( arr_size, arr);
	arr_size= 0;
      }
    }
    else 
      break;

    if ( strempty(dum2,dum2) )
      dum= 0;
    else
      dum= dum2;
  }
  //cout << "\n------------\nend of while, dum=[" << dum << "]"; //debug

  if ( !support && arr_size < 2)
    return false;

  if ( dum ) {
    if ( ! strskip(dum,"(",dum) ) 
      return false;

    if ( ! (support == 0 && arr_size == 2) )
      return false;

    if ( str2val(dum, num, dum) != 1 )
      return false;

    if ( num < 2)
      return false;

    if ( ! strskip(dum,")",dum) ) 
      return false;

    if ( !strempty(dum,dum) )
      return false;

    min= arr[0];
    max= arr[1];
    if (min >= max)
      return false;
    return true;
  }

  append_array(arr_size,arr);
  min= support[0];
  max= support[ num-1 ];
  for (int i=1; i< num; i++) //the sequence of values must be strictly increasing
    if (support[i-1] >= support[i])
      return false;
  return true;
}

LatticeOpt::LatticeOpt() {
  DIM= 0;
  KACZMARZ= false;
}

float * concat_arrays( int len, const float * arr, int len2, const float * arr2) {
  int new_len= len + len2;

  if (new_len < 0)
    return 0;

  float *new_arr= new float[new_len];
  
  if ( new_len <= len && len > 0 ) {
    memcpy( (void*)new_arr,(void*)arr, sizeof(float) * new_len);
    return new_arr;
  }

  if ( new_len <= len2 && len2 > 0 ) {
    memcpy( (void*)new_arr,(void*)arr2, sizeof(float) * new_len);
    return new_arr;
  }

  if ( len > 0) 
    memcpy( (void*)new_arr,(void*)arr, sizeof(float) * len);
  
  if ( len2 > 0)
    memcpy( (void*)(new_arr+len),(void*)arr2, sizeof(float) * len2);

  return new_arr;
}

bool LatticeOpt::read(istream & in) {
  const int buffer_size= 512;
  char buffer[buffer_size];
  const char * dum;
  bool got_dim_num= false;
  bool got_dim[MAX_DIM];
  bool got_beg_options= false;
  for (int i=0; i<MAX_DIM; i++)
    got_dim[i]= false;

  while (in) {
    in.getline(buffer,buffer_size);

    //cout << "\nbuffer= ["<< buffer << "]" << flush;

    if (!got_beg_options){
      if ( strempty(buffer) ){
        continue;
      }
      else if ( strskip(buffer,"<options>",dum) ) {
        if ( strempty(dum,dum) )
          got_beg_options= true;
        else
          return false;
        continue;
      }
      else {
        return false;
      }
    }

    //here: got_beg_options == true
    
    if ( strskip(buffer,KACZMARZ_STR,dum) ) {
      if ( !str2val(dum,KACZMARZ,dum) )
	return false;
      continue;
    }

    if (!got_dim_num) {
      if ( strskip(buffer,"#") || strempty(buffer) ){
        continue;
      }
      else if ( strskip(buffer,DIM_STR,dum) && strspace(dum) ) {
        int tmp= str2val(dum,DIM,dum);
        if ( DIM <= 0 || DIM >  MAX_DIM || !tmp || !strempty(dum) ) {
          return false;
        }
        got_dim_num= true;
        continue;
      } 
      else{
        return false;
      }
    }
    
    //here: got_dim_num == true
    if ( strskip(buffer,"#") || strempty(buffer) )
      continue;
    if ( strskip(buffer,"</options>",dum) ) {
      if ( !strempty(dum) ) 
	return false;

      break;
    }
    else {
      bool got_input= false;
      for (int i=0; i< DIM; i++) {
	char dum_buf[ strlen(DIM_STR) +3 ];
	sprintf(dum_buf,"%s_%d",DIM_STR,i);
	if ( strskip(buffer,dum_buf,dum) ) {
	  if ( got_dim[i] ) {
	    ERROR_OUT << "\ndim " << i << " defined more then once";
	    return false;
	  }

	  const char * dum2;
	  if ( strskip(dum,"[",dum2) ) {
	    dum= dum2;
	    bool has_end= false;
	    while (true) {
	      if ( strskip(dum,"]",dum2) ) {
		dum= dum2;
		has_end= true;
		break;
	      }
	      if ( strskip(dum,"C",dum2) ) {
		dum= dum2;
		slices[i].circular= true;
		continue;
	      }
	      break;
	    }
	    if (!has_end)
	      return false;
	  }

	  if ( ! slices[i].read(dum) ) {
	    ERROR_OUT << "\ndim " << i << " wrong slice " << dum;
	    return false;
	  }

	  got_dim[i]= true;
	  got_input= true;
	  break;
	}
      }
      if (got_input)
	continue;
    }
    return false;
  }

  if (!got_dim_num)
    return false;

  for (int i=0; i<DIM; i++)
    if ( !got_dim[i]) return false;

  write(cout);

  return true;
}

void LatticeOpt::write(ostream & out) const {
  out << "\n<options>"
      << "\n" << DIM_STR << " " << DIM;
  for (int i=0; i<DIM; i++) {
    const LatticeSlice & s= slices[i];
    out << "\n" << DIM_STR << "_" << i;
    if (s.circular)
      out << " [C]";
    if ( ! s.support ) 
      out << " " << s.min << " " << s.max << " (" << s.num << ")";
    else 
      for (int j=0; j< s.num; j++) 
	out << " " << s.support[j];
  }
  out << "\n" << KACZMARZ_STR << " " << KACZMARZ;
  out << "\n</options>";
}

int LatticeOpt::get_tab_size() const {
  if (DIM <=0 ) 
    return 0;

  int res= 1;
  for (int i=0; i < DIM; i++)
    res *= slices[i].num;
  return res;
}
/*****************************************************************************/

void LatticeMap::body_write(ostream & out) const {
  if ( options.DIM>0 && tab ) {
    out << "\n<body>\n";
    //out.write( (const void*) tab, tab_size * sizeof(float) );
    out.write( (const char*) tab, tab_size * sizeof(float) );
    out << "\n</body>";
  }
}

bool LatticeMap::body_read(istream & in) {
  const int buffer_size= 512;
  char buffer[buffer_size];
  const char * dum;
  //bool got_end_body= false;

#if 0
  while (in) {
    in.getline(buffer,buffer_size);
    if ( strskip(buffer,"") )
      continue;
    else if ( strskip(buffer,"<body>",dum) ) {
      if ( !strempty(dum) )
	return false;
      break;
    }
    else 
      return false;
  }
#endif
  cout << "\nreading table" << flush;
  in.read((char*)tab, tab_size * sizeof(float));
  
  while (in) {
    in.getline(buffer,buffer_size);
    if ( strempty(buffer) )
      continue;
    else if ( strskip(buffer,"</body>",dum) ) {
      if ( !strempty(dum) )
	return false;
      return true;
    }
    else
      break;
  }
  return false;
}

void LatticeMap::point_to_index(const float *p, int & idx) const {
  int m_idx[options.DIM];
  point_to_mindex(p,m_idx);
  mindex_to_index(m_idx,idx);
}

void LatticeMap::point_to_mindex(const float *p, int * m_idx) const {
  for (int i=0; i< options.DIM; i++) {
    float d= p[i];
    const LatticeSlice & s= options.slices[i];
    if ( d < s.min ) 
      m_idx[i]= 0;
    else if ( d > s.max ) 
      m_idx[i]= s.num-1;
    else {
      if ( ! s.support ) {
	d -=  s.min;
	d *= (s.num-1);
	d /= (s.max - s.min); 
	//round to next int
	float d_ceil=  ceil(d);
	float d_floor= floor(d);
	if ( fabs(d_ceil -d) < fabs(d_floor -d) ) //choose that index, so that the lattice point has nearest euclidean distance to it
	  m_idx[i]= int( d_ceil );
	else 
	  m_idx[i]= int( d_floor );
      }
      else {
	int min_idx= 0; 
	int max_idx= s.num-1;
	int num_idx= s.num;
	//invariant:  support[min_idx] <= d < support[max_idx] (or d <= support[num-1])
	while (num_idx > 2) { //binary search
	  int dum_idx= min_idx + num_idx /2; // min_idx < dum_idx < max_idx
	  if ( d < s.support[ dum_idx ] )
	    max_idx= dum_idx;
	  else
	    min_idx= dum_idx;

	  num_idx= max_idx - min_idx + 1;
	}
	if (min_idx + 1 != max_idx) {//debug
	  cout << "\nWRONG  min_idx= " << min_idx << "  max_idx= " << max_idx;
	  exit(1);
	}

	float left_bound= s.support[min_idx];
	float right_bound= s.support[max_idx];
	if ( right_bound - d > d - left_bound ) //nearer to the left support point
	  m_idx[i]= min_idx;
	else 
	  m_idx[i]= max_idx;
#if 0
	cout << "\n coord in dim " << i << " , " 
	     << " support[" << min_idx << "]= " << s.support[ min_idx]
	     << " <= " << d << " < "
	     << " support[" << max_idx << "]= " << s.support[ max_idx]
	     << " res_idx= " << m_idx[i];
#endif
      }
    }
    /* if circular => identify the right entry with the left most entry 
       (right most entry is not used if this dim is circular)
    */
    if ( s.circular && m_idx[i] == s.num -1 ) 
      m_idx[i]= 0;
  }
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
bool LatticeMap::point_to_mindex_quad_coord(const float *p, int * m_idx, float *quad_coord) const {
  int res= true;
  for (int i=0; i< options.DIM; i++) {
    float d= p[i];
    const LatticeSlice & s= options.slices[i];
    if ( d < s.min ) {
      m_idx[i]= 0;
      res= false;
      return res; // if the result is false, the mindex is not needed (so far, maybe it will change)
    }
    else if ( d > s.max ) {
      m_idx[i]= s.num-1;
      res= false;
      return res;
    }
    else {
      if ( ! s.support ) { //the are no support points, so take equidistant subdivision
	d -=  s.min;
	d *= (s.num-1);
	d /= (s.max - s.min); 
	//round to next int
	float d_floor= floor(d);
	quad_coord[i]= d-d_floor; 
	m_idx[i]= int( d_floor );
	if (m_idx[i] + 1 == s.num)  {//correct for values on the outer edge!
	  m_idx[i]= s.num-2;
	  quad_coord[i]= 1.0;
	}
      }
      else { //consider the subdivision defined by support, 
	int min_idx= 0; 
	int max_idx= s.num-1;
	int num_idx= s.num;
	//invariant:  support[min_idx] <= d < support[max_idx] (or d <= support[num-1])
	while (num_idx > 2) { //binary search
	  int dum_idx= min_idx + num_idx /2; // min_idx < dum_idx < max_idx
	  if ( d < s.support[ dum_idx ] )
	    max_idx= dum_idx;
	  else
	    min_idx= dum_idx;

	  num_idx= max_idx - min_idx + 1;
	}
	if (min_idx + 1 != max_idx) {//debug
	  ERROR_OUT << "\nWRONG  min_idx= " << min_idx << "  max_idx= " << max_idx;
	  exit(1);
	}
	  
	m_idx[i]= min_idx;
	quad_coord[i]= (d - s.support[min_idx])/ ( s.support[max_idx]- s.support[min_idx]);

#if 0
	cout << "\n coord in dim " << i << " , " 
	     << " support[" << min_idx << "]= " << s.support[ min_idx]
	     << " <= " << d << " < "
	     << " support[" << max_idx << "]= " << s.support[ max_idx]
	     << " quad_coord= " << quad_coord[i];
#endif
      }
    }
  }
  return res;
}

void LatticeMap::compute_barycentric_coord_and_simplex_path(float * quad_coord,int * path_idx) const {
  const int DIM= options.DIM;
  for (int i=0; i<DIM; i++) 
    path_idx[i]= i;

  //sort with respect to quad_coord coordinates
  //invariat: before_sort_quad_coord[i]= after_sort_quad_coord[ after_sort_path_idx[i] ];
  for (int i=0; i<DIM; i++) {
    int idx_of_largest= i;
    for (int j=i; j<DIM; j++) 
      if ( quad_coord[j] > quad_coord[idx_of_largest] )
	idx_of_largest= j;
    if (idx_of_largest != i) { //swap entries in quad_coord and path_idx resp.
      float f_tmp= quad_coord[i]; 
      quad_coord[i]= quad_coord[idx_of_largest]; 
      quad_coord[idx_of_largest]= f_tmp;

      int i_tmp= path_idx[i];
      path_idx[i]= path_idx[idx_of_largest];
      path_idx[idx_of_largest]= i_tmp;
    }
  }
  //now compute the barycentric coordinate out of the quad_coordinates with respect to the 
  //simplex given by the path_idx
  quad_coord[DIM]= 0.0;
  for (int b= DIM; b> 0; b--) {
    quad_coord[b]= quad_coord[b-1] - quad_coord[b];
  }
  quad_coord[0]= 1.0 - quad_coord[0];
}

bool LatticeMap::point_to_mindex_barycentric_coord_and_simplex_path(const float * x, 
								    int * m_idx, 
								    float * bar_coord, 
								    int * path_idx) const {
  bool res= point_to_mindex_quad_coord(x,m_idx,bar_coord);
  if (res) 
    compute_barycentric_coord_and_simplex_path(bar_coord, path_idx);
  return res;
}

bool LatticeMap::point_to_indexes_of_barycentric_coord(const float * x, int * bar_idx, float * bar_coord) const {
  const int DIM= options.DIM;
  int m_idx[DIM];
  int * path_idx= bar_idx+1;
  int res= point_to_mindex_barycentric_coord_and_simplex_path(x,m_idx,bar_coord,path_idx);
  if (!res)
    return res;
  //now the path_idx is in bar_idx+1, it remains to transform it to the indexes of the barycentric points
  //compute the indexes of the barycentric points
  int idx;
  mindex_to_index(m_idx,idx);
  bar_idx[0]= idx;

  for (int d= 0; d< DIM; d++) { 
    int i_d= path_idx[d]; //in this dimension we go now
    m_idx[ i_d ]+= 1;
#if 1
    /* begin of <circularity overhead> */
    const LatticeSlice & s= options.slices[i_d];
    /* if circular => identify the right entry with the left most entry 
       (right most entry is not used if this dim is circular)
    */
    if ( s.circular && m_idx[i_d] == s.num -1 ) 
      m_idx[i_d]= 0;
    /* end of <circularity overhead> */
#endif
    mindex_to_index(m_idx,idx);
    bar_idx[ d+1 ] = idx; // there is no conflict in writing the value here, because the value of path_idx[d]= bar_idx[d+1] is no longer needed.
  }  
  return res;
}

float LatticeMap::kaczmarz_get_value_at(const float * x) const {
  const int DIM= options.DIM;
  float bar_coord[DIM+1];
  int   bar_idx[DIM+1];
  bool res;

  res= point_to_indexes_of_barycentric_coord(x,bar_idx,bar_coord);
  if (!res)
    return kronecker_get_value_at(x); 

  float val= 0.0;
  //compute the weighted sum of values at all barycentric coordinate points
  for (int b= 0; b< DIM+1; b++) {
    int idx= bar_idx[b];
    val += tab[idx] * bar_coord[ b ];
  }

  return val;
}

void LatticeMap::kaczmarz_add_sample_point(const float * x, float target)  {
  const int DIM= options.DIM;
  float bar_coord[DIM+1];
  int   bar_idx[DIM+1];
  bool res;

  res= point_to_indexes_of_barycentric_coord(x,bar_idx,bar_coord);
  if (!res) {
    //kronecker_add_sample_point(x,target); 
    return; 
  }

  float old_val= 0.0;
  //compute the weighted sum of values at all barycentric coordinate points
  for (int b= 0; b< DIM+1; b++) {
    int idx= bar_idx[b];
    old_val += tab[idx] * bar_coord[ b ];
  }
  //here the value at point x is in old_val
  /* now use kaczmarz update  
       f_{n+1}(x_i)= f_n(x_i) + b(i)/(b'b) * [ target - old_val ]
     where b(i) is the barycentric coordinate with respect to simplex point i
     and b'b= sum_{i=0,...,DIM} b(i)^2  the scalar product <b,b>
  */
  float bar_scalar= 0.0;
  for (int b=0; b < DIM+1; b++) 
    bar_scalar += bar_coord[b]*bar_coord[b];
  
#if 0
  cout << "\n---" 
       << "\ntarget= " << target
       << "\nold_val= " << old_val
       << "\nbar_scalar= " << bar_scalar;
  for (int b=0; b < DIM+1; b++) {
    cout << "\n bar[" << b <<"]= " << bar_coord[b] << " idx= " << bar_idx[b];
  }
#endif
  
  for (int b=0; b < DIM+1; b++) {
    int idx= bar_idx[b];
    //cout << "\ntab[" << idx << "]= " << tab[idx] << " -> ";
    tab[idx] += bar_coord[b]/ bar_scalar * ( target - old_val);
    //tab[idx] += 1.0/(float(DIM+1)) * ( target - old_val);  // just a test for lim n->0
   
    //cout << tab[idx];
  }
}
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
void LatticeMap::mindex_to_index(const int *m_idx, int & idx) const {
  int res= 0;
  for( int i=0; i< options.DIM-1; i++) {
    res += m_idx[i];
    res *= options.slices[i+1].num;
  }
  res += m_idx[options.DIM-1];
  idx= res;
}

void LatticeMap::index_to_mindex(int idx,int *m_idx) const {
  for( int i= options.DIM-1; i>0; i--) {
    m_idx[i]= idx % options.slices[i].num;
    idx = idx / options.slices[i].num;
  }
  m_idx[0]= idx;
}

void LatticeMap::set_matrix_point(const int * m_idx, float target) {
  int idx;
  mindex_to_index(m_idx,idx);
  tab[idx]= target;
}

float LatticeMap::get_matrix_point(const int * m_idx) const {
  int idx;
  mindex_to_index(m_idx,idx);
  return tab[idx];
}

void LatticeMap::add_sample_point(const float * x, float target) { 
  if (options.KACZMARZ)
    kaczmarz_add_sample_point(x,target);
  else
    kronecker_add_sample_point(x,target);
}

void LatticeMap::kronecker_add_sample_point(const float * x, float target) { 
  int idx;
  point_to_index(x,idx);
  tab[idx]= target;
}

float LatticeMap::get_value_at(const float * x) const {
  if (options.KACZMARZ)
    return kaczmarz_get_value_at(x);
  return kronecker_get_value_at(x);
}

float LatticeMap::kronecker_get_value_at(const float * x) const {
  int idx;
  point_to_index(x,idx);
  return tab[idx];
}

bool LatticeMap::save(const char * fname) const {
  ofstream out(fname);
  if (!out) {
    ERROR_OUT << "\nsave: cannot open file " << fname;
    return false;
  }
  out << "<lattice_map_2>";
  options.write(out);
  body_write(out);

  out << "\n</lattice_map_2>";
  out.close();
  return true;
}

bool LatticeMap::load(const char * fname) {
  const int buffer_size= 512;
  char buffer[buffer_size];
  const char * dum;
  bool got_header= false;
  bool got_end_header= false;
  bool got_junk= false;
  bool got_body= false;
  empty_lattice();

  ifstream in(fname);
  if (!in) {
    ERROR_OUT << "\nload: cannot open file " << fname;
    return false;
  }

  while (in) {
    in.getline(buffer,buffer_size);
    if ( strempty(buffer) )
      continue;
    else if ( strskip(buffer,"<lattice_map_2>",dum) ) {
      if ( !strempty(dum) ) 
	return false;
      got_header= true;
      break;
    } 
    else if ( strskip(buffer,"<lattice_map>",dum) ) {
      ERROR_OUT << "\nlatticee map file '" << fname << "' has old version (1)"
		<< "\nchange "
		<< "\n    <lattice_map>                   -> <lattice_map_2>"
		<< "\n    </lattice_map>                  -> </lattice_map_2>"
		<< "\n    DIM_<int> <float> <float> <int> -> DIM_<int> <float> <float> '('<int>')'"
		<< "\nto convert to the newest version (2) of lattice map"
		<< "\n\nnew feature in version 2:"
		<< "\nby"
		<< "\n    DIM_<int> <float> <float> ... <float>"
		<< "\nwith a strictly increasing sequence, you can define not equidinstant"
		<< "\npartitions of an axis."
		<< "\n";
      return false;
    } 
    else
      break;
  }

  if (!got_header) {
    ERROR_OUT << "\nload: wrong file format " << fname;
    in.close();
    return false;
  }

  if (!options.read(in)) {
    ERROR_OUT << "\nload: errors while reading options from " << fname;
    in.close();
    return false;
  }
    
  if (options.DIM>0) {
    int dum= options.get_tab_size();
    if (tab && dum != tab_size ) {
      delete[] tab;
      tab= 0;
    }
    tab_size= dum;
    if (!tab)
      tab= new float[tab_size];
  }

  while (in) {
   in.getline(buffer,buffer_size);
    if ( strempty(buffer) )
      continue;
    else if ( strskip(buffer,"</lattice_map_2>",dum) ){
      if ( !strempty(dum) )
	return false;
      got_end_header= true;
      break;
    }
    else if ( strskip(buffer,"<body>",dum) ) {
      if (!strempty(dum)) 
	return false;
      got_body= true;
      if (body_read(in))
	break;
      else
	return false;
    }
    in.close();
    ERROR_OUT << "\nload: corrupted end of " << fname;
    return false;
  }

  if (!got_body)  //init table to zero
    set_all_entries(0.0);


  if (!got_end_header)
  while (in) {
    in.getline(buffer,buffer_size);
    if ( strempty(buffer) )
      continue;
    else if ( strskip(buffer,"</lattice_map_2>",dum) ){
      if (!strempty(dum))
	return false;
      got_end_header= true;
      break;
    }
    else
      break;
  }

  while (in) {
    in.getline(buffer,buffer_size);
    if ( strempty(buffer) )
      continue;
    else 
      got_junk= true;
  }

  in.close();
  if (!got_end_header || got_junk) {
    ERROR_OUT << "\nload: corrupted end of " << fname;
    return false;
  }
  
  return true;
}

void LatticeMap::set_all_entries(float entry) {
  for (int i=0; i < tab_size; i++)
    tab[i]= entry;
}

bool LatticeMap::set_dim_and_resolution(int d, int resolution) {
  if (d < 0 || d > options.MAX_DIM)
    return false;
  
  options.DIM= d;
  for (int i=0; i< d; i++) {
    LatticeSlice & s= options.slices[i];
    if ( s.support && s.num != resolution ) {
      WARNING_OUT << "\ndim " << i << ": using new resolution deletes support points";
      delete[] s.support;
      s.support= 0;
    }
    s.num= resolution; 
  }

  int dum= options.get_tab_size();

  if (tab) {
    if (dum==tab_size) //no resize is needed!
      return true;
    delete[] tab;
  }
  tab_size= dum;
  tab= new float[tab_size];
  return true;
}

bool LatticeMap::set_dim_and_resolutions(int d, int * resolutions) {
  if (d < 0 || d > options.MAX_DIM)
    return false;
  
  options.DIM= d;
  for (int i=0; i< d; i++) {
    LatticeSlice & s= options.slices[i];
    if ( s.support && s.num != resolutions[i] ) {
      WARNING_OUT << "\ndim " << i << ": using new resolution deletes support points";
      delete[] s.support;
      s.support= 0;
    }
    s.num= resolutions[i];
  }
  int dum= options.get_tab_size();

  if (tab) {
    if (dum==tab_size) //no resize is needed!
      return true;
    delete[] tab;
  }
  tab_size= dum;
  tab= new float[tab_size];
  return true;
}

bool LatticeMap::set_bound_in_all_dim(float min, float max) {
  if ( min >= max)
    return false;

  for (int i= 0; i< options.DIM; i++) {
    LatticeSlice & s= options.slices[i];
    s.min= min;
    s.max= max;
    if ( s.support) {
      WARNING_OUT << "\ndim " << i << ": using bound deletes support points";
      delete[] s.support;
      s.support= 0;
    }
  }
  return true;
}

bool LatticeMap::set_bound_in_dim(int d, float min, float max) {
  if (d < 0 || d > options.MAX_DIM)
    return false;
  if (min >= max)
    return false;
  LatticeSlice & s= options.slices[d];
  s.min= min;
  s.max= max;
  if ( s.support) {
    WARNING_OUT << "\ndim " << d << ": using bound deletes support points";
    delete[] s.support;
    s.support= 0;
  }

  return true;
}

bool LatticeMap::get_bound_in_dim(int d, float & min, float & max) const {
  if (d < 0 || d > options.MAX_DIM)
    return false;
  min= options.slices[d].min;
  max= options.slices[d].max;
  return true;
}

bool LatticeMap::set_support_in_dim(int d, int num, const float * sup) {
  if (d < 0 || d > options.MAX_DIM )
    return false;

  if ( num < 2 ) {
    ERROR_OUT << "\ntoo few  support points  provided (" << num << ")";
    return false;
  }

  for (int i=0; i < num-1; i++) {
    if ( sup[i] >= sup[i+1] ) {
      ERROR_OUT << "\nsupport points should be in increasing order";
      return false;
    }
  }

  LatticeSlice & s= options.slices[d];

  if (num != s.num) { //resizing of the table is necessary
    s.num= num;
    int dum= options.get_tab_size();
    if (tab && dum != tab_size ) {
      delete[] tab;
      tab= 0;
    }
    tab_size= dum;
    if (!tab)
      tab= new float[tab_size];

    if ( s.support ) {
      delete[] s.support;
      s.support= 0;
    }
  }
  if ( ! s.support )
    s.support= new float[s.num];
  for (int i=0; i < s.num; i++)
    s.support[i]= sup[i];

  s.min= s.support[0];
  s.max= s.support[s.num -1];

  return true;
}

bool LatticeMap::set_circular_in_dim(int d, bool f) {
  if (d < 0 || d > options.MAX_DIM)
    return false;
  options.slices[d].circular= f;
  return true;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

#if 0

#include <fstream.h>
#include "plot.h"
#include "randomize.h"

void test_linear_1d_fun() {
  LatticeMap lmp;

  lmp.set_dim_and_resolution(1,6);
  const float sup[6]= { 0, 0.5, 1 , 3, 6, 10};
  lmp.set_bound_in_dim(0,0,10);
  lmp.set_circular_in_dim(0,true);
  //lmp.set_support_in_dim(0,6, sup);

  //lmp.save("bla.lmp");
  //return;
  Slice1d s1(0,128,0,10);   
  float xxx;
  lmp.set_kaczmarz();
  //lmp.set_kronecker();
  const int num= 21;
  for (int i=0; i<5; i++) {
    for (int j=0; j<num; j++) {
      xxx= 0.0 + j * 10.0/float(num-1);
      xxx= drand_uniform_in_0_1() * 10.0;
      float ttt= xxx;
      lmp.add_sample_point(&xxx, ttt);
      //cout << "\n (x= (" << xxx << ")->" << ttt << " " << lmp.get_value_at(&xxx);
      if ( true && j % 4 == 0 ) {
	char buf[100];
	sprintf(buf, "test %d %d %f",i,j,xxx);
	gnuplot_1d_slice_of_function(buf,lmp,s1,1,&xxx);
	int d;
	cin >> d;
      }
    }

  }

  //cout << "\ntab="; lmp.show(); cout << "\n<<<<\n";
#if 1
  gnuplot_1d_slice_of_function("test",lmp,s1,1,&xxx);

  lmp.set_kronecker();

  gnuplot_1d_slice_of_function("test",lmp,s1,1,&xxx);
#endif
}

void test_linear_2d_fun() {
  LatticeMap lmp;

  lmp.set_dim_and_resolution(2,6);
  const float sup[6]= { 0, 0.5, 1 , 3, 6, 10};
  const float sup2[3]= { 0, 3, 10};
  lmp.set_bound_in_dim(0,0,10);
  //lmp.set_support_in_dim(0,6, sup);
  lmp.set_support_in_dim(0,3, sup2);
  lmp.set_bound_in_dim(1,0,10);

  float xxx[2];
  xxx[0]= 0.0;
  lmp.set_kaczmarz();
  for (int j=0; j<6; j++) {
    xxx[1]= 0.0;
    for (int i=0; i<6; i++) {
      float ttt= 2.0 + (i+4) % 6;
      ttt= 1.0;
      ttt= i+j;
      lmp.add_sample_point(xxx, ttt);
      cout << "\n (x,y)= (" << xxx[0] <<"," << xxx[1] << ")->" << ttt << " " << lmp.get_value_at(xxx);
      xxx[1]+= 2.0;
    }
    xxx[0]+= 2.0;
  }

  lmp.save("bla.lmp");
  LatticeMap lmp2;
  lmp2.load("bla.lmp");
  lmp2.save("bla2.lmp");
  //cout << "\ntab="; lmp.show(); cout << "\n<<<<\n";
  /*
  Slice1d s1(0,64,0,10);   Slice1d s2(1,64,0,10);

  gnuplot_2d_slice_of_function("test",lmp,s1,s2,2,xxx);

  lmp.set_kronecker();

  gnuplot_2d_slice_of_function("test",lmp,s1,s2,2,xxx);
  */
}

int main(int argc, char ** argv) {
  LatticeMap lmp; lmp.load("bla.lmp"); lmp.save("bla2.lmp"); return 0;
  test_linear_1d_fun();
  //test_linear_2d_fun();
  return 0;
#if 0
  xxx[0]= 1; xxx[1]= 1;
  xxx[0]= 9; xxx[1]= 9;
  lmp.add_sample_point(xxx, 13);
  plot_2d_slice_of_function("test3",lmp,s1,s2,2,xxx);
  lmp.show();
  return 0;
  LatticeMap lm;
  cout << "\nBEGIN" << endl;
  if (argc>1) {
    if (!lm.load(argv[1])) {
      ERROR_OUT << "\n ERRORS WHILE READING FROM FILE " << argv[1];
    }
    else {
      float x[10];
      for (int i=0; i<10; i++)
	x[i]= 0;
      
      lm.add_sample_point(x,10);
      if (argc>2) {
	lm.show();
	lm.save(argv[2]);
	LatticeMap lm2;
	lm2.load(argv[2]);
	lm2.show();
      }
    }
  }

  return 1;
  int res= 1;
  for (int i=0; i < lm.options.DIM; i++)
    res *= lm.options.slices[i].num;

  int x[lm.options.DIM];
  for (int i=0; i<res; i++) {
    lm.index_to_mindex(i,x);
    
    cout << "\n" << i << " ->";
    for (int k=0; k< lm.options.DIM; k++)
      cout << " " << x[k];
    int dum;
    lm.mindex_to_index(x,dum);
    cout << " -> " << dum;
  }

  while(1) {
    float x[2];
    int midx[2];
    cout <<"\n>";
    scanf("%f %f",x,x+1);
    lm.point_to_mindex(x,midx);
    for (int k=0; k< lm.options.DIM; k++)
      cout << " " << midx[k] <<"(" << x[k] << ")";
  }
  
#endif
  return 1;
}
#endif
