#ifndef _PFILTER_H_
#define _PFILTER_H_

#include <iostream>
#include "Vector.h"
//#include "distributions.h"
#include "normal_dist2d.h"

static const int DIM_STATE= 2;

typedef double State[3];//[DIM_STATE+1];

class ParticleSet {
 public:
  int size;
  State * tab;

  ParticleSet(int s) {
    size= s;
    tab= new State[size];
  }

  ~ParticleSet() {
    if (tab)
      delete[] tab;
  }

  double sum_of_weights() {
    double sum= 0.0;
    for (int i=0; i<size; i++) 
      sum += tab[i][0];
    return sum;
  }

  double sum_of_square_weights() {
    double sum= 0.0;
    for (int i=0; i<size; i++) 
      sum += tab[i][0]*tab[i][0];
    return sum;
  }
  
  void compute_mean(State mean) {
    for (int d= 0; d<= DIM_STATE; d++) 
      mean[d]= 0.0;

    for (int i=0; i< size; i++) {
      State & state= tab[i];

      mean[0] += state[0];
      for (int d= 1; d<= DIM_STATE; d++) 
	mean[d] += state[0]* state[d];
    }

    for (int d= 1; d<= DIM_STATE; d++) 
      mean[d] /= mean[0];
  }

  void show(std::ostream & out);

};

class ParticleFilter {
 private:  
  double tmp_arr[2];
  
  ParticleSet pset;
  
  //double lower[] = {-0.1,-0.1};
  //double upper[] = {0.1,0.1};

  //Uniform_distribution odometry_distr(2, lower, upper);
  //Gaussian_distribution odometry_distr(2, 0.1, 0.1);
  //Gaussian_distribution sensor_distr(2, 0.5, 0.5);
  Normal_distribution_2d odometry_distr;
  Normal_distribution_2d sensor_distr;

  bool init(ParticleSet & pset, Vector initial_values);

  int get_particle_index_for_random_number(ParticleSet &pset, double rand);

  bool sample_particles(ParticleSet & pset);

  bool integrate_odometry(ParticleSet &pset, Vector movement);

  bool weight_particles(ParticleSet & pset, Vector sensed_position);

 public:

  ParticleFilter() : pset(50), odometry_distr(0.01, 0.01, 0.0), sensor_distr(0.25, 0.25, 0.0) {};
  ~ParticleFilter() {};
  
  void update_with_sensed_movement(Vector sensed_movement);
  
  void update_with_sensed_position(Vector sensed_position);
  
  int update_with_sensed_markers(int num, Vector const *markers, double const *dist, Vector sensed_position);

  Vector get_position();

  void init_pset(Vector position);

};

#endif
