#ifndef _EPISODE_PROCESSING_NEURAL_NETWORK_TRAINER_H_
#define _EPISODE_PROCESSING_NEURAL_NETWORK_TRAINER_H_

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include <string>
#include <sstream>
#include <vector>

#include "ConfigReader.h"
#include "n++.h"


#define ERROR_OUT                         std::cout
#define FILE_NAME_LENGTH                  100
#define NN_MAXIMAL_NUMBER_OF_PARAMETERS   5
#define BALL_SPEED_MAX                    2.7
#define FIELD_BORDER_X                    52.5
#define FIELD_BORDER_Y                    34.0
#define MAX_NUMBER_OF_STATE_VARIABLES     50
#define rand_0_1()                        drand48()

using namespace std;

class EpisodeProcessingNeuralNetworkTrainer
{
  public:
    struct State
    {
      public:
        int   ivSize;
        float ivFeatures[MAX_NUMBER_OF_STATE_VARIABLES];
        State();
        State( const State & state );
        State( const vector<float> & vectorOfFloats );
    };
  
  private:
    char                  ivEpisodeFileName[FILE_NAME_LENGTH];
    Net                 * ivpNeuralNetwork;
    int                   ivNegativeEpisodesCounter;
    int                   ivPositiveEpisodesCounter;
    int                   ivUseSuccessfulEpisodesOnly;
    float                 ivContinuedEpisodeReuseShare;
    int                   ivNetworkTrainingMethod;
    char                  ivNeuralNetworkFileName[FILE_NAME_LENGTH];
    int                   ivNumberOfLearningEpochs;
    int                   ivNumberOfInputNeurons;
    int                   ivNumberOfHiddenNeurons;
    int                   ivNumberOfOutputNeurons;
    float                 ivScaleA, ivScaleB;
    vector<float>         ivVNeuralNetworkParameters;
    vector< pair<State,float> >   ivVNeuralNetTrainingExamples;
    vector< pair<State,float> >   ivVPositiveNeuralNetTrainingExamples;
    vector< pair<State,float> >   ivVNegativeNeuralNetTrainingExamples;
    
    void                  addTrainingExamples
                          ( int posneg,
                            vector< pair<State,float> > & currentEpisode );
    float                 calculateNeuralNetworkError();
    void                  calculateTargetValues
                          ( vector< pair<State,float> > & currentEpisode );
    bool                  createNeuralNetwork();
    bool                  readInSingleEpisode( FILE * episodeFileHandle,
                                               bool   useThisEpisode);
    bool                  readTrainingDataFromFile
                                             ( char  * fname,
                                               float   episodeUsageProbability );
    void                  scaleTargetValues();
    void                  setNeuralNetInputFromState(const State & s);
    void                  trainNeuralNetwork();
    
  protected:
  
  public:
    EpisodeProcessingNeuralNetworkTrainer();
    ~EpisodeProcessingNeuralNetworkTrainer();
    
    bool  createTrainingData();
    void  doTraining();
    bool  init( char * configFileName );
    void  setEpisodeFileName( char * episodeFileName );  
  
};

#endif
