//#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "pfilter.h"
#include "macro_msg.h"
//#include "draw.h"
//#include "distributions.h"
//#include "valuereader.h"
#include <math.h>
#include <sys/time.h>

/*
void ParticleFilter::show(ostream & out) {
  out << "size= " << size;
  for (int i=0; i<size; i++) {
    State & s= tab[i];
    out << "\np" << i << ")= (w=" << s[0];
    for (int d=0; d<DIM_STATE; d++) {
      out << ", " << s[d+1];
    } 
    out << ")";
  }
}
*/

bool ParticleFilter::init(ParticleSet & pset, Vector initial_values) {
  double weight= 1.0/ double(pset.size);
  for (int i=0; i<pset.size; i++) {
    State & s= pset.tab[i];
    s[1] = initial_values.getX();
    s[2] = initial_values.getY();
    s[0]= weight;
  }
  //initialize the random number generator only onece during initialization
  timeval tval;
  if (gettimeofday(&tval,NULL))
    ERROR_OUT << " :  something wrong with time mesurement";
  srand48((unsigned long)tval.tv_usec);
  //srand48 ( time(NULL) );
  return true;
}


int ParticleFilter::get_particle_index_for_random_number(ParticleSet &pset, double rand) {
  double sum = 0.0;
  for (int i = 0; i < pset.size; i++) {
    sum += pset.tab[i][0];
    if (sum >= rand) return i;
  }

  return 0;
}

bool ParticleFilter::sample_particles(ParticleSet & pset) {
  double rand;
  int index;

  double sum = 0.0;
  for (int i = 0; i < pset.size; i++) {
    sum += pset.tab[i][0];
  }
  if (sum < 0.95) ERROR_OUT << "FEHLER in sample_particles, weight sum less than 1\n";

  ParticleSet p_tmp(pset.size);
  for (int i = 0; i < pset.size; i++) {
    for (int j = 0; j < DIM_STATE + 1; j++) {
      p_tmp.tab[i][j] = pset.tab[i][j];
    }
  }

  for (int i = 0; i < pset.size; i++) {
    rand = drand48();
    index = get_particle_index_for_random_number(p_tmp, rand);
    for (int j = 0; j < DIM_STATE + 1; j++) {
      pset.tab[i][j] = p_tmp.tab[index][j];
      continue;
    }
  }
  return true;
}

bool ParticleFilter::integrate_odometry(ParticleSet &pset, Vector movement) {
  double temp[2];
  for (int i = 0; i < pset.size; i++) {
    temp[0] = movement.getX() + pset.tab[i][1];
    temp[1] = movement.getY() + pset.tab[i][2];
    odometry_distr(pset.tab[i][1], pset.tab[i][2], temp[0], temp[1]);
  }
  return true;
}

bool ParticleFilter::weight_particles(ParticleSet & pset, Vector sensed_position) {
  double sum_of_weights= 0.0;
  for (int i=0; i< pset.size; i++) {
    State & state = pset.tab[i];
    state[0] = sensor_distr.pdf(sensed_position.getX(), sensed_position.getY(), state[1], state[2]);
    sum_of_weights += state[0];
  }

  //normalize weights
  for (int i=0; i< pset.size; i++) {
    State & state = pset.tab[i];
    state[0] /= sum_of_weights;
  }
  return true;
}


//double lower[] = {-0.1,-0.1};
//double upper[] = {0.1,0.1};

//Gaussian_distribution odometry_distr(2, 0.1, 0.1);
//Uniform_distribution odometry_distr(2, lower, upper);
//Gaussian_distribution sensor_distr(2, 0.5, 0.5);
//ParticleSet pset(100);


void ParticleFilter::update_with_sensed_movement(Vector sensed_movement) {
  sample_particles(pset);
  integrate_odometry(pset, sensed_movement);
}

void ParticleFilter::update_with_sensed_position(Vector sensed_position) {
  State mean;
  pset.compute_mean(mean);
  double sqr_dist = pow(mean[1] - sensed_position.getX(), 2) + pow(mean[2] - sensed_position.getY(), 2);
  
  if (sqr_dist > 2.5) {
    ERROR_OUT << "ERROR in ParticleSet: sensed position too far from mean value ----->>>>> reinitialize\n";
    init(pset, sensed_position);
    return;
  }

  if (isnan(mean[1]) || isnan(mean[2])) {
    ERROR_OUT << "ERROR in ParticleSet: mean value not a number ----->>>>> reinitialize\n";
    init(pset, sensed_position);
    return;
  }

  weight_particles(pset, sensed_position);

}

int ParticleFilter::update_with_sensed_markers(int num, Vector const *markers, double const *dist, Vector sensed_position) {

  State mean;
  pset.compute_mean(mean);
  double sqr_dist = pow(mean[1] - sensed_position.getX(), 2) + pow(mean[2] - sensed_position.getY(), 2);
  int ret = 1; 

  if (sqr_dist > 0.3) {
    pset.tab[0][0] = 0.05;
    pset.tab[0][1] = sensed_position.getX();
    pset.tab[0][2] = sensed_position.getY();
    ret = 3;
  }

  if (isnan(mean[1]) || isnan(mean[2])) {
    ERROR_OUT << "ERROR in ParticleSet_new: mean value not a number ----->>>>> reinitialize\n";
    init(pset, sensed_position);
    return 1;
  }

  double sum_of_weights= 0.0;
  bool use_particle;
  int num_used_particles = 0;
  double sqr_dist_to_marker;

  double sqr_upper_dist[num];
  double sqr_lower_dist[num];
  for (int i = 0; i < num; i++) {
    sqr_upper_dist[i] = (dist[i] + 0.05) * 1.005012521;
    sqr_upper_dist[i] *= sqr_upper_dist[i];
    sqr_lower_dist[i] = (dist[i] - 0.05) / 1.005012521;
    sqr_lower_dist[i] *= sqr_lower_dist[i];
  }

  for (int i=0; i< pset.size; i++) {
    State & state = pset.tab[i];
    use_particle = true;
    for (int j = 0; j < num; j++) {
      sqr_dist_to_marker = pow(state[1] - markers[j].getX(), 2) + pow(state[2] - markers[j].getY(), 2);
      if (sqr_upper_dist[j] < sqr_dist_to_marker ||
	  sqr_lower_dist[j] > sqr_dist_to_marker) {
	use_particle = false;
	break;
      }
    }
    if (use_particle == false) state[0] = 0.01;
    else {
      state[0] = 1.0;
      num_used_particles++;
    }
    sum_of_weights += state[0];
  }

  /*
  if (num_used_particles < 1) {
    ERROR_OUT << "ERROR in ParticleSet_new: not enough particles used ----->>>>> reinitialize " << num_used_particles<< "\n";
    init(pset, sensed_position);
    return;
    }*/

  //normalize weights
  for (int i=0; i< pset.size; i++) {
    State & state = pset.tab[i];
    state[0] /= sum_of_weights;
  }
  return ret;
}

Vector ParticleFilter::get_position() {
  State mean;
  pset.compute_mean(mean);
  return Vector(mean[1], mean[2]);
}


void ParticleFilter::init_pset(Vector position) {
  //std::cout << "init_pset aufgerufen\n";
  init(pset, position);

}

/*
#define DRAW 0
int main(int argc, char ** argv) {


  DrawEngine DDD;

  ValueReader vr;
  vr.append_from_command_line(argc,argv);


  bool draw= true;
  long int seed= 323203;
  vr.get("d",draw);
  vr.get("seed",seed);

  if (draw) {
    DDD.init();
    DDD.set_empty();
    DDD.set_visual_area(25,0,100,80);
  }

  //Statistics::Random::seed(1213214);
  Statistics::Random::seed(seed);


  double action[] = {0,1};


  Gaussian_distribution odometry_distr(2, 0.1, 0.1);
  Gaussian_distribution sensor_distr(2, 0.25, 0.25);

  Gaussian_distribution real_odometry_distr(2, 0.1, 0.1);
  Gaussian_distribution real_sensor_distr(2, 0.25, 0.25);


  State state = {1.0, 0.0, 0.0};
  State sensed_state = {1.0, 0.0, 0.0};
  State movement = {1.0, 4.0, 1.0};
  State sensed_movement = {1.0, 4.0, 1.0};
  
  ParticleSet pset(200);
  init(pset, state);

  double error_sum = 0.0;
  
  for (int i=0; i<100; i++) {
    State mean;
    pset.compute_mean(mean);

    double error= 0;
    for (int d= 1; d<= DIM_STATE; d++) {
      double dum= state[d] - mean[d];
      error += dum*dum;
    }
    error= sqrt(error);
    error_sum += error;
    cout << "\n" << i << " error= " << error;

    state[1] += movement[1];
    state[2] += movement[2];
    //std::cout << "state is " << state[1] << " " << state[2] << "\n";
    real_odometry_distr.random(sensed_movement+1, movement+1);
    //std::cout << "sensed movement is " << sensed_movement[1] << " " << sensed_movement[2] << "\n";
    real_sensor_distr.random(sensed_state+1, state+1);
    //std::cout << "sensed state is " << sensed_state[1] << " " << sensed_state[2] << "\n";

    sample_particles(pset);
    integrate_odometry(pset, odometry_distr, sensed_movement+1);
    weight_particles(pset, sensor_distr, sensed_state+1);


    if (draw) {
      DDD.set_empty();
      //DDD.set_visual_area(25,0,100,80);
      DDD.draw(pset);
      DDD.draw_state(state);
      sleep(1);
    }

    if (i%20 == 0) std::cout << "sum of errors after " << i << " steps is " << error_sum;

  }


  if (draw) {
    DDD.draw(pset);
  }
}
*/
