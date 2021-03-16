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
#ifndef _LATTICE_MAP_H_
#define _LATTICE_MAP_H_

#include <iostream>
#include <fstream>
#include <stdlib.h>

struct LatticeSlice {
  LatticeSlice() { num= 0; support= 0; circular= false; }
  LatticeSlice(float minimum, float maximum, int size) { min= minimum; max= maximum; num= size; circular= false; }

  bool read(const char * str);
  float min,max;
  int num;
  // if support != 0, the it has 'num' elements, with 
  // support[0]= min < ...  support[i] ...< support[j] < ... < support[num-1]= max, if 0 < i < j < num-1 >= 1
  float * support;
  bool circular;
  protected:
  void append_array(int len, const float * arr);
};

struct LatticeOpt {
  static const int MAX_DIM= 10;
  LatticeOpt();
  int DIM;
  bool KACZMARZ;
  LatticeSlice slices[MAX_DIM];

  static const char * const DIM_STR;//= "DIM";
  static const char * const KACZMARZ_STR;//= "KACZMARZ";

  bool read(std::istream &);
  void write(std::ostream &) const;
  
  int get_tab_size() const;
};

/** For kronecker and kaczmarz update take a look at the paper:
    @InProceedings{Par:Adaptive,
    author =       {S. Pareigis},
    title =        "{Adaptive choice of grid and time in reinforcement learning}",
    booktitle =    {NIPS},
    OPTcrossref =  {},
    OPTkey =       {},
    OPTpages =     {7},
    year =         {1997}
    }
    For computation of the simplex in a Kuhn-triangulation see:
    @InProceedings{,
    author =       {R. Munos and and A. Moore},
    title =        "{Variable resolution discretization for high-accuracy solutions of optimimal control problems}",
    booktitle =    {IJCAI}
    }
    or better "Variable Resolution Discretization in Optimal Control" (24S.) by same 
    authors but unknown source ;-(

    Important notice about how a simplex can be represented.
    We use the representation mindex + simplex_path.
    mindex is the multiindex of a point in the table and simplex_path is a 
    permutation of {0,...,DIM-1}. From this two array you can attain a simplex 
    in the following manner:

    0. Point: mindex
    1. Point: set mindex[ simplex_path[0] ] + 1, mindex 
    2. Point: set mindex[ simplex_path[1] ] + 1
    ...
    D Point: set mindex[ simplex_path[DIM-1] ] + 1

    put it another way:

    (m_0, m_1, ... , m_k, ..., m_l , ... , m_{D-1}) where k= simplex_path[0],where l= simplex_path[1];

    so you get form 0. point to 1. point

    (m_0, m_1, ... , m_k+1, ..., m_l , ... , m_{D-1}) 

    and with 

    (m_0, m_1, ... , m_k+1, ..., m_l+1 , ... , m_{D-1}) 
      
    to 2. point, and so on. The D. point is always equal to 

    (m_0+1, m_1+1, ... , m_k+1, ..., m_l+1 , ... , m_{D-1}+1) 

    so every single index is increased by 1.

    -------------------

    remark for usage of circularity in one or more dimension(s).

    if in one dimension circularity is defined, then nothing changes of 
    the point of memory allocation. Only the most right point in this
    dimension (with index 'slices[i].num-1') is always identified
    with the left most point. Using kronecker updates it is very easy
    implemented. Using kaczmarz it is a little tricky to find the right
    place of this conversion, but it works.

    switching circular mode on/off:
    considering memory it is absolutely secure do do it. Anyway 
    if you switch between these two modes, you loose some information,
    because in circular mode you work with one representative of an
    equivalence class, and switching to non circular mode doesn't 
    also update the other point in your equivalence class (not at the 
    moment of writing these line Sep. 2001)
 */
class LatticeMap {
  //void wbin(std::ostream &) const { }
  //void rbin(std::istream &) { }
  void empty_lattice() { }
  bool body_read(std::istream &);
  void body_write(std::ostream &) const;
  float *tab;
  int tab_size;
  void INIT() { tab= 0; tab_size= 0;}

  LatticeOpt options;
 public:

  LatticeMap() { INIT(); }
  //LatticeMap(const LatticeOpt & opt) { }
  LatticeMap(const char * fname)  { INIT(); if (!load(fname)) exit(1); }
  ~LatticeMap() { }
  void add_sample_point(const float * x, float target);
  float get_value_at(const float * x) const;
  int dim() const { return options.DIM; }

  void set_matrix_point(const int * m_idx, float target);
  float get_matrix_point(const int * m_idx) const;
  int get_matrix_size_in_dim(int d) { return options.slices[d].num; }

  void set_array_point(int idx, float target) { tab[idx]= target; }
  float get_array_point(int idx) { return tab[idx]; }
  int get_array_size() { return tab_size; }

  void set_kronecker() { options.KACZMARZ= false; }
  void set_kaczmarz() { options.KACZMARZ= true; }
  void set_all_entries(float entry);
  bool set_dim_and_resolution( int d, int resolution);
  bool set_dim_and_resolutions( int d, int * resolutions);

  bool set_bound_in_all_dim(float min, float max);
  bool set_bound_in_dim( int d, float min, float max);
  bool get_bound_in_dim( int d, float & min, float & max) const;

  bool set_support_in_dim( int d, int num, const float * sup);

  bool set_circular_in_dim( int d, bool f) ;

  bool save(const char * fname) const;
  bool load(const char * fname);
 protected:
  void kronecker_add_sample_point(const float *x, float target);
  void kaczmarz_add_sample_point(const float *x, float target);
  float kronecker_get_value_at(const float * x) const;
  float kaczmarz_get_value_at(const float *x) const;
  /** important quad_coord must have dim= DIM+1 to transform the quad_coord 
      into barycentric coordinates. 
      Precondition: 0 <= quad_coord[i] <= 1 , i=0...DIM-1
      Postcondition: 0 <= quad_coord[i] <= 1 , i=0...DIM, and sum of all is 1
                     0 <= path_idx[i] < DIM, i=0...DIM-1 gives the simplex!
		     path_idx is a permutation of {0,...,DIM-1}
  */
  void compute_barycentric_coord_and_simplex_path(float * quad_coord,int * path_idx) const;
  /** should be called after 
      f= point_to_mindex_quad_coord(const float *p, int * m_idx,float *quad_coord) const;
      m_idx contains the result after f(p,m_idx,quad_coord);
      bar_coord expects as input the quad_coord after f(p,m_idx,quad_coord); 
      note: the required dim for quad_coord is DIM, and for bar_coord it is DIM+1,
            so the bar_coord[DIM] value doesn't need to be specified!
      path_idx doesn't need to be initilized.
  */
  bool point_to_mindex_barycentric_coord_and_simplex_path(const float * x, int * m_idx, float * bar_coord, int * path_idx) const;
  /** THIS method is used in 
      kaczmarz_get_value_at and 
      kaczmarz_add_sample_point methods. It uses (implicitly)
      
      calls:
         point_to_mindex_barycentric_coord_and_simplex_path
         calls:
	    point_to_mindex_quad_coord
	    //and with values computed by point_to_mindex_quad_coord
            compute_barycentric_coord_and_simplex_path
      
      Output: computes the indexes of all barycentric coordinates in the Kuhn-Simplex
      around the given point (it also comutes the corresponding barycentric
      coordinates)
      dim of bar_idx and bar_coord should be DIM+1 */
  bool point_to_indexes_of_barycentric_coord(const float * x, int * bar_idx, float * bar_coord) const;
  /** just combines point_to_mindex with mindex_to_index */
  void point_to_index(const float *p, int & idx) const;
  /** remark on point_to_mindex and point_to_mindex_quad_coord:
      the resulting m_idx are not identical!
      point_to_mindex looks for the nearest (in euclidean norm) point on the 
      grid, and delivers it's mindex;
      point_to_mindex_quad_coord looks for the nearest (in euclidean norm) 
      point which all coordinates are smaller than the coordinates of the given 
      point itself. 
      The corresponding mindex and the coordinates with respect this point are
      computed. 
      
      point_to_mindex always computes a result in m_idx. It is the mindex of 
      the nearest point in the grid, even if the given point is in far distance
      from the grid. Maybe this method also should return a false value to 
      indicate if in some direction the point is further away from the grid 
      than a half cell size (corresponding to this dimension).
  */
  void point_to_mindex(const float *p, int * m_idx) const;
  /** result is false if the point is not in the grid's (min,max) area,  
      if the result is false, the m_idx does not contain any information (it 
      can be changed in the future if needed, but normally we use the nearest 
      euclidean point if the point isn't in the grid).
  */
  bool point_to_mindex_quad_coord(const float *p, int * m_idx,float *quad_coord) const;
  /** compute the index in the table auf of the multi index of a point */
  void mindex_to_index(const int *m_idx, int & idx) const;
  /** this direction will probably be never needed, but just to make the bijective 
      map fully implemented ;-), (this feature is useful to check if the 
      point_to_mindex method is really one to one!) */
  void index_to_mindex(int idx,int *m_idx) const;

  void show() {
    if (tab) 
      for (int i=0; i<tab_size; i++)
	std::cout << "\ntab[" << i << "]= " << tab[i];
  }
};

/*****************************************************************************/
#endif
