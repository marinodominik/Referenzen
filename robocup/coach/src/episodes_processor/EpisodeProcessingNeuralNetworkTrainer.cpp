#include "EpisodeProcessingNeuralNetworkTrainer.h"

using namespace std;

//############################################################################
// IMPLEMENTATION OF INNER CLASS STATE
//############################################################################
EpisodeProcessingNeuralNetworkTrainer::State::State()
{
  ivSize = 0;
}

EpisodeProcessingNeuralNetworkTrainer::State::State( const State & state )
{
  ivSize = state.ivSize;
  if (ivSize > MAX_NUMBER_OF_STATE_VARIABLES)
  {
    ERROR_OUT<<"ERROR: State: Feature "<<"vector size exceeds maximum."<<endl;
    ivSize = MAX_NUMBER_OF_STATE_VARIABLES;
  }
  for (int i=0; i<ivSize; i++)
    ivFeatures[i] = state.ivFeatures[i];
}

EpisodeProcessingNeuralNetworkTrainer
  ::State::State( const vector<float> & vectorOfFloats )
{
  ivSize = (int)vectorOfFloats.size();
  if (ivSize > MAX_NUMBER_OF_STATE_VARIABLES)
  {
    ERROR_OUT<<"ERROR: State: Feature "<<"vector size exceeds maximum."<<endl;
    ivSize = MAX_NUMBER_OF_STATE_VARIABLES;
  }
  for (int i=0; i<ivSize; i++)
    ivFeatures[i] = vectorOfFloats[i];
}


//############################################################################
// IMPLEMENTATION OF CLASS EPISODEPROCESSINGNEURALNETWORKTRAINER
//############################################################################

//============================================================================
// CONSTRUCTORS / DESTRUCTORS
//============================================================================
EpisodeProcessingNeuralNetworkTrainer::EpisodeProcessingNeuralNetworkTrainer()
{
  ivpNeuralNetwork = NULL;
  ivNegativeEpisodesCounter   = 0;
  ivPositiveEpisodesCounter   = 0;
  ivUseSuccessfulEpisodesOnly  = 0;
  ivContinuedEpisodeReuseShare = 0.0;
}
EpisodeProcessingNeuralNetworkTrainer::~EpisodeProcessingNeuralNetworkTrainer()
{
  if (ivpNeuralNetwork) delete ivpNeuralNetwork;
}

//============================================================================
// STATIC METHODS
//============================================================================


//============================================================================
// NON-STATIC METHODS
//============================================================================

//============================================================================
// addTrainingExamples
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer
  ::addTrainingExamples( int posneg,
                         vector< pair<State,float> > & currentEpisode )
{
  if (posneg == 1) 
    ivPositiveEpisodesCounter ++ ;
  else
    ivNegativeEpisodesCounter ++ ;
  for ( unsigned int i=0; i < currentEpisode.size(); i++)
  {
    pair<State,float> trainingExample;
    trainingExample.first  = currentEpisode[i].first;
    trainingExample.second = currentEpisode[i].second;
    if (posneg == 1) //POS
    {
      ivVPositiveNeuralNetTrainingExamples.push_back( trainingExample );
    }
    else //NEG
    {
      ivVNegativeNeuralNetTrainingExamples.push_back( trainingExample );
    }
  }
}

//============================================================================
// doTraining
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer::doTraining()
{
  this->scaleTargetValues();
  this->trainNeuralNetwork();
  
  //write out summary
  ERROR_OUT<<"****************** SUMMARY ******************"<<endl;
  ERROR_OUT<<"  ivScaleA = "<<ivScaleA<<"    ivScaleB = "<<ivScaleB<<endl;
  ERROR_OUT<<"  positive episodes considered: "<<ivPositiveEpisodesCounter
    <<" => "<<ivVPositiveNeuralNetTrainingExamples.size()<<" TEs"<<endl;
  ERROR_OUT<<"  negative episodes considered: "<<ivNegativeEpisodesCounter
    <<" => "<<ivVNegativeNeuralNetTrainingExamples.size()<<" TEs"<<endl;
}

//============================================================================
// calculateNeuralNetworkError
//============================================================================
/**
 * Diese Methode ist insofern recht rechenintensiv, als sie eine komplette
 * Iteration ueber alle gesammeltend Trainingsadaten und jeweils
 * eine Vorwaertspropagation durch das Netz durchfuehrt, um so den insgesamten
 * Fehler des Netzes zu errechnen.
 */ 
float
EpisodeProcessingNeuralNetworkTrainer::calculateNeuralNetworkError()
{
  int numOfTrainingExamples = ivVNeuralNetTrainingExamples.size();
  float error = 0.0, pError = 0.0, mError = 0.0;
  for (int e=0; e<numOfTrainingExamples; e++)
  {
    float target = ivVNeuralNetTrainingExamples[e].second;
    this->setNeuralNetInputFromState( ivVNeuralNetTrainingExamples[e].first );
    ivpNeuralNetwork->forward_pass( ivpNeuralNetwork->in_vec,
                                    ivpNeuralNetwork->out_vec );
    //debug output
    if (rand_0_1()<0.0001 || e<40 || target<0.3)
    {
      stringstream debString;
      debString<<"Feature Vector (l="
        <<ivVNeuralNetTrainingExamples[e].first.ivSize<<"): ";
      for (int i=0; i<ivNumberOfInputNeurons; i++)
      {
        debString<<ivVNeuralNetTrainingExamples[e].first.ivFeatures[i]<<" / ";
      }
      ERROR_OUT<<"INFO: NN Discrepancy: target="<<target
        <<"\tnetOut="<<ivpNeuralNetwork->out_vec[0]
        <<"\tdelta="<<target - ivpNeuralNetwork->out_vec[0]<<endl;
      ERROR_OUT<<" for: "<<debString.str().c_str()<<endl;
    } //end of debug output
    
    error += (target - ivpNeuralNetwork->out_vec[0])
             * (target - ivpNeuralNetwork->out_vec[0]);
    if (ivpNeuralNetwork->out_vec[0] - target > 0.0)
      pError += ivpNeuralNetwork->out_vec[0] - target;
    else
      mError += ivpNeuralNetwork->out_vec[0] - target;
  }
  ERROR_OUT<<"INFO: pError = "<<pError<<", mError = "<<mError<<", total="
    <<(error / (float)numOfTrainingExamples)<<endl;

  return error / (float)numOfTrainingExamples;
}

//============================================================================
// calculateTargetValues
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer
  ::calculateTargetValues( vector< pair<State,float> > & currentEpisode )
{
  if (currentEpisode.size() == 0) return;
  float summedReward = 0.0;
  for ( int i = (int)currentEpisode.size() - 1;
        i >= 0;
        i -- )
  {
    float immediateReward = currentEpisode[i].second;
    currentEpisode[i].second 
      = currentEpisode[i].second + summedReward;
    summedReward += immediateReward;
ERROR_OUT<<" target = "<<currentEpisode[i].second <<endl;
  }
}

//============================================================================
// createNeuralNetwork
//============================================================================
bool
EpisodeProcessingNeuralNetworkTrainer::createNeuralNetwork()
{
  ivpNeuralNetwork = new Net();
  
  if ( ivpNeuralNetwork->load_net( ivNeuralNetworkFileName ) != FILE_ERROR )  
  {
    ERROR_OUT<<"EpisodeProcessingNeuralNetworkTrainer: Warning: Neural "
      <<"network "<<ivNeuralNetworkFileName<<" already exists."
      <<endl<<flush;
    delete ivpNeuralNetwork;
    ivpNeuralNetwork = new Net();
  }
  
  int nodesPerLayer[3];
  nodesPerLayer[0] = ivNumberOfInputNeurons;
  nodesPerLayer[1] = ivNumberOfHiddenNeurons;
  nodesPerLayer[2] = ivNumberOfOutputNeurons;
  float learnParams[NN_MAXIMAL_NUMBER_OF_PARAMETERS];
  int backPropagationOrRProp = ivNetworkTrainingMethod; //0==BP, 1==RPROP
  //default parameters
  if ( backPropagationOrRProp == 0 ) //BP
  {  
    learnParams[0] = 0.01;     //Lernrate
    learnParams[1] = 0.0;      //Momentum
  }
  else //RPROP
  {
    learnParams[0] = 0.001; //delta null
    learnParams[1] = 10.0;  //delta max 
  }
  for (int i=2; i<NN_MAXIMAL_NUMBER_OF_PARAMETERS; i++)
    learnParams[i] = 0.0;  
  //read parameters
  for ( unsigned int i=0; 
           i < NN_MAXIMAL_NUMBER_OF_PARAMETERS
        && i < ivVNeuralNetworkParameters.size();
        i ++ )
    learnParams[i] = ivVNeuralNetworkParameters[i];
  //create net structure
  ivpNeuralNetwork->create_layers(3, nodesPerLayer);
  ivpNeuralNetwork->connect_layers();
  //alle Gewichte initial zufaellig zwischen -0.5 und 0.5
  ivpNeuralNetwork->init_weights(0, 0.5); 
  ivpNeuralNetwork->set_update_f(backPropagationOrRProp, learnParams); 
  //scaling of inputs
  ivpNeuralNetwork->set_input_scaling( 1, 1, 1.0/FIELD_BORDER_X, 0.0);
  ivpNeuralNetwork->set_input_scaling( 2, 1, 1.0/FIELD_BORDER_Y, 0.0);
  ivpNeuralNetwork->set_input_scaling( 3, 1, 1.0/BALL_SPEED_MAX, 0.0);
  ivpNeuralNetwork->set_input_scaling( 4, 1, 1.0/BALL_SPEED_MAX, 0.0);
  for ( int i=5; i<ivNumberOfInputNeurons; i++)
  { //default scaling
    if (i%2 == 1) //ungerade inputs (zaehlung ab 1): x-koordinaten
      ivpNeuralNetwork->set_input_scaling( i, 1, 1.0/FIELD_BORDER_X, 0.0);
    if (i%2 == 0) //gerade inputs (zaehlung ab 1): y-koordinaten
      ivpNeuralNetwork->set_input_scaling( i, 1, 1.0/FIELD_BORDER_Y, 0.0);
  }
  if (ivpNeuralNetwork->save_net(ivNeuralNetworkFileName) != FILE_ERROR)
  {
    ERROR_OUT<<"EpisodeProcessingNeuralNetworkTrainer: "
      <<"A new neural network has been created "
      <<"and saved ("<<ivNeuralNetworkFileName<<")."<<endl<<flush;
  }
}

//============================================================================
// createTrainingData
//============================================================================
bool
EpisodeProcessingNeuralNetworkTrainer::createTrainingData()
{
  bool returnValue = true;
  returnValue &= readTrainingDataFromFile( ivEpisodeFileName, 1.0 );
  
  if ( ivContinuedEpisodeReuseShare > 0.0 )
  {
    ERROR_OUT<<"Invoke a system call ..."<<endl;
    int ret
      = system("/bin/ls -1 *_episodes.txt > episodeFiles.txt");
    if (ret==-1) ERROR_OUT<<"Error executing system call."<<endl;
    else ERROR_OUT<<"Command successfully executed"<<endl;
    FILE * fileNameHandle = fopen( "episodeFiles.txt", "r" );
    if ( fileNameHandle == NULL )
    {
      ERROR_OUT<<"System call did not succeed. Cannot open episodeFiles.txt"
        <<endl<<flush;
      return false;
    }
    char buffer[100]; buffer[0] = '\0';
    do
    {
      fgets( buffer, 100, fileNameHandle );
      ERROR_OUT<<"Read a line from episodeFiles.txt ... "<<buffer<<endl;
      if ( buffer != NULL && strlen(buffer) > 0 && strlen(buffer) < 100
           && feof(fileNameHandle) == false )
      {
        if (strncmp( buffer, "/bin/ls:", 8) == 0)
        {
          ERROR_OUT<<"System call succeeded, but there are no files with "
            <<"stored episodes."<<endl<<flush;
          return false;
        }
        buffer[strlen(buffer)-1] = '\0';
        ERROR_OUT<<"Start reading episodes from file "<<buffer<<endl;
        returnValue &= readTrainingDataFromFile( buffer,
                                                 ivContinuedEpisodeReuseShare );
      }
    }
    while ( feof(fileNameHandle) == false );
    ERROR_OUT<<"End of file of episodeFiles.txt reached. "<<buffer<<endl;
    fclose( fileNameHandle );
  }
  else
  {
    ERROR_OUT<<"Do not reuse older episodes! (ivContinuedEpisodeReuseShare="
      <<ivContinuedEpisodeReuseShare<<")"<<endl;
  }
}

//============================================================================
// init
//============================================================================
bool
EpisodeProcessingNeuralNetworkTrainer::init( char * configFileName )
{
  Tribots::ConfigReader cr(0);
  cr.append_from_file( configFileName );
  string s;
  
  cr.get("EpisodeProcessingNeuralNetworkTrainer::episodeFileName", s);
  strcpy( ivEpisodeFileName, s.c_str());
  cr.get("EpisodeProcessingNeuralNetworkTrainer::neuralNetworkFileName", s);
  strcpy( ivNeuralNetworkFileName, s.c_str());
  cr.get("EpisodeProcessingNeuralNetworkTrainer::useSuccessfulEpisodesOnly", 
          ivUseSuccessfulEpisodesOnly);
  cr.get("EpisodeProcessingNeuralNetworkTrainer::continuedEpisodeReuseShare", 
          ivContinuedEpisodeReuseShare);
  cr.get("EpisodeProcessingNeuralNetworkTrainer::numberOfLearningEpochs", 
          ivNumberOfLearningEpochs);
  cr.get("EpisodeProcessingNeuralNetworkTrainer::networkTopologyInputNeurons", 
          ivNumberOfInputNeurons );
  cr.get("EpisodeProcessingNeuralNetworkTrainer::networkTopologyHiddenNeurons", 
          ivNumberOfHiddenNeurons );
  cr.get("EpisodeProcessingNeuralNetworkTrainer::networkTopologyOutputNeurons", 
          ivNumberOfOutputNeurons );
  cr.get("EpisodeProcessingNeuralNetworkTrainer::networkTrainingMethod", 
          ivNetworkTrainingMethod );
  cr.get("EpisodeProcessingNeuralNetworkTrainer::networkParameterVector", 
          ivVNeuralNetworkParameters );
          
  this->createNeuralNetwork();
  return true;
}

//============================================================================
// readInSingleEpisode
//============================================================================
bool
EpisodeProcessingNeuralNetworkTrainer::readInSingleEpisode     
                                       (FILE * episodeFileHandle,
                                        bool   useThisEpisode)
{
  char buffer[2000];
  //consider episode length
  int episodeLength = 0;
  if (    fgets( buffer, 2000, episodeFileHandle ) 
       && strlen( buffer ) > 7 )
    episodeLength = atoi( buffer + 7 );
  else
  {
    ERROR_OUT<<"Error reading episode length"<<endl<<flush;
    return false;
  }
  //consider episode classification
  int episodeClassification = -1; //not usable
  if (    fgets( buffer, 2000, episodeFileHandle )
       && strlen( buffer ) > 14 )
    episodeClassification = atoi( buffer + 14 );
  else
  {
    ERROR_OUT<<"Error reading episode classification"<<endl<<flush;
    return false;
  }
  //consider ignorable episodes
  if ( episodeClassification < 0 )
  {
    while (    fgets(buffer, 2000, episodeFileHandle )
            && strncmp( buffer, "END_EPISODE", 11) != 0 )
    {  }
    ERROR_OUT<<"Episode has been ignored!"<<endl<<flush;
    return false;
  }
  //read in current episode
  vector< pair<State,float> > currentEpisode;
  for (int i=0; i<episodeLength; i++)
  {
    //read time stamp
    char * readResult = fgets( buffer, 2000, episodeFileHandle );
    int currentTimeStep = -1;
    if ( readResult && strlen( buffer ) > 9 )
      currentTimeStep = atoi( buffer + 9 );
    if ( currentTimeStep != i )
    {
      ERROR_OUT<<"Error reading in time step information."<<endl<<flush;
      exit(0);
    }
    //read state information
    State currentState;
    readResult = fgets( buffer, 2000, episodeFileHandle );
    vector<float> stateValues;
    if ( readResult )
    {
      char * tokenizerPtr = strtok( buffer, " \n");
      int cnt=0;
      while ( tokenizerPtr != NULL)
      {
        stateValues.push_back( atof(tokenizerPtr) );
        tokenizerPtr = strtok(NULL, " \n");
        cnt++;
      }
      if ( (int)stateValues.size() != ivNumberOfInputNeurons )
      {
        ERROR_OUT<<"Error reading in state information (b): "
          <<(int)stateValues.size()<<" vs. "<<ivNumberOfInputNeurons
          <<endl<<flush;
        exit(0);
      }
      currentState = State( stateValues );
    }
    else
    {
      ERROR_OUT<<"Error reading in state information (a)."<<endl<<flush;
      exit(0);
    }
    //read in immediate reward
    readResult = fgets( buffer, 2000, episodeFileHandle );
    float currentReward = 0.0;
    if ( readResult )
    {
      currentReward = atof( buffer );
    }
    else
    {
      ERROR_OUT<<"Error reading in reward information."<<endl<<flush;
      exit(0);
    }
    //extend the current episode
    pair<State,float> currentTrainingExample;
    currentTrainingExample.first  = currentState;
    currentTrainingExample.second = currentReward;
    currentEpisode.push_back( currentTrainingExample );
  }
  //check for end of episode
  if (   fgets( buffer, 2000, episodeFileHandle ) != NULL
      && strncmp( buffer, "END_EPISODE", 11) == 0 )
  {
    //end of episode has been found
    if (useThisEpisode)
    {
      this->calculateTargetValues( currentEpisode );
      this->addTrainingExamples( episodeClassification, currentEpisode );
    }
    return true;
  }
  else
  {
    ERROR_OUT<<"Error detecting the end of the episode."<<endl<<flush;
    exit(0);
  }
  
}

//============================================================================
// readTrainingDataFromFile
//============================================================================
bool
EpisodeProcessingNeuralNetworkTrainer::readTrainingDataFromFile
                                       ( char  * fname,
                                         float   episodeUsageProbability )
{
  FILE * episodeFileHandle = fopen( fname, "r" ); //read
  if ( episodeFileHandle == NULL )
  {
    ERROR_OUT<<"Could not open episode file "<<fname<<" (length="<<
      strlen(fname)<<")"<<endl<<flush;
    return false;
  }
  
  //loop reading in episodes
  char buffer[100];
  char * readResult = NULL;
  int episodeCounter = 0;
  do
  {
    float randomNumber = rand_0_1();
    bool useThisEpisode = false;
    if (randomNumber < episodeUsageProbability ) useThisEpisode = true;
    char * readResult = fgets( buffer, 100, episodeFileHandle );
    if (   (readResult == NULL && feof(episodeFileHandle) )
        || strncmp( buffer, "START_EPISODE", 13 ) != 0 )
    {
      ERROR_OUT<<"Finished reading from "<<fname<<endl;
      break;
    }
    else
    {
      if ( this->readInSingleEpisode( episodeFileHandle, useThisEpisode ) )
        episodeCounter ++ ;
      ERROR_OUT<<"Episode read. Number of usable episodes: "<<episodeCounter
        <<", #te="<<ivVPositiveNeuralNetTrainingExamples.size()
                   +ivVNegativeNeuralNetTrainingExamples.size()<<endl;
    }
  }
  while ( feof(episodeFileHandle) == false );

  fclose( episodeFileHandle );
  return true;
}

//============================================================================
// scaleTargetValues
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer::scaleTargetValues()
{
  float targetSum = 0.0, targetMax = 0.0, targetMin = INT_MAX;
  for (unsigned int i=0; i<ivVPositiveNeuralNetTrainingExamples.size(); i++)
  {
    targetSum += ivVPositiveNeuralNetTrainingExamples[i].second;
    if (ivVPositiveNeuralNetTrainingExamples[i].second > targetMax)
      targetMax = ivVPositiveNeuralNetTrainingExamples[i].second;
    if (ivVPositiveNeuralNetTrainingExamples[i].second < targetMin)
      targetMin = ivVPositiveNeuralNetTrainingExamples[i].second;
  }
  float targetAvg 
    = targetSum / (float)ivVPositiveNeuralNetTrainingExamples.size();
    
  //calculate scaling values
  //TODO: scaling mechanism must be improved!
  //      Die folgenden beiden Gleichungen ergeben sich bei Aufstellen
  //      und Loesen eines entsprechenden linearen Gleichungssystems.
  float netMin = 0.1,
        netMax = 0.9,
        netDelta = netMax - netMin,
        targetDelta = targetMax - targetMin;
  ivScaleA = targetDelta / netDelta;
  ivScaleB = targetMax - ivScaleA * netMax;
  //analog: ivScaleB = minTarget - ivScaleA * netMin;
  for (unsigned int i=0; i<ivVPositiveNeuralNetTrainingExamples.size(); i++)
  {
    float targetValue = ivVPositiveNeuralNetTrainingExamples[i].second;
    targetValue = (targetValue - ivScaleB) / ivScaleA;
    //too large target values may distort the net.
    //[Bei der aktuellen einfachen Skalierungsmethodik koennen die
    //folgenden Faelle ueberhaupt nicht auftreten.]
    if (targetValue > netMax)
    {
      targetValue = ((targetValue-netMax)/100.0) + netMax;
      if (targetValue > 0.98) targetValue = 0.98;
    }
    //too small values may distort the net, too.
    if (targetValue < 0.0)
      targetValue = 0.0;
      
    pair<State,float> scaledTrainingExample;
    scaledTrainingExample.first = ivVPositiveNeuralNetTrainingExamples[i].first;
    scaledTrainingExample.second= targetValue;
    ivVNeuralNetTrainingExamples.push_back( scaledTrainingExample );
    ERROR_OUT<<"POSITIVE TEs: "<<i<<": "<<targetValue<<endl;
  }

  //Iteration ueber die negativen Trainingsbeispiele
  if (ivUseSuccessfulEpisodesOnly == 0)
    for (unsigned int i=0; i<ivVNegativeNeuralNetTrainingExamples.size(); i++)
    {
      float targetValue = ivVNegativeNeuralNetTrainingExamples[i].second;
      targetValue = (targetValue - ivScaleB) / ivScaleA;
      //too large target values may distort the net.
      //[Bei der aktuellen einfachen Skalierungsmethodik koennen die
      //folgenden Faelle ueberhaupt nicht auftreten.]
      if (targetValue > netMax)
      {
        targetValue = ((targetValue-netMax)/10.0) + netMax;
        if (targetValue > 0.98) targetValue = 0.98;
      }
      //too small values may distort the net, too.
      if (targetValue < 0.0)
        targetValue = 0.0;
        
      pair<State,float> scaledTrainingExample;
      scaledTrainingExample.first = ivVNegativeNeuralNetTrainingExamples[i].first;
      scaledTrainingExample.second= targetValue;
      ivVNeuralNetTrainingExamples.push_back( scaledTrainingExample );
      ERROR_OUT<<"NEGATIVE TEs: "<<i<<": "<<targetValue<<endl;
    }
  
  ERROR_OUT<<"Scaling Summary for Positive Episodes: "<<endl;  
  ERROR_OUT<<"targetMax="<<targetMax<<" targetMin="<<targetMin<<" targetAvg="<<targetAvg<<endl;
  ERROR_OUT<<"ivScaleA = "<<ivScaleA<<" ivScaleB = "<<ivScaleB<<endl;
}

//============================================================================
// setEpisodeFileName
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer
  ::setEpisodeFileName( char * episodeFileName )
{
  strncpy( ivEpisodeFileName, episodeFileName, FILE_NAME_LENGTH );
}  

//============================================================================
// setNeuralNetInputFromState
//============================================================================
/**
 * Diese Methode nimmt einen State (dies ist eine struct-Datenstruktur, die
 * im Wesentlichen einen float-Vektor sowie eine Groessenangabe ueber den
 * Floatvektor enthaelt) entgegen und setzt auf dessen Basis die Netzeingabe.
 */ 
void 
EpisodeProcessingNeuralNetworkTrainer
  ::setNeuralNetInputFromState(const State & s)
{
  for (int i=0; i<ivNumberOfInputNeurons; i++)
  {
    ivpNeuralNetwork->in_vec[i] = s.ivFeatures[i];
  }
}

//============================================================================
// trainNeuralNetwork
//============================================================================
void
EpisodeProcessingNeuralNetworkTrainer::trainNeuralNetwork()
{
  char helpingString[100];
  float e1, e2;
  float initialError = -1.0;
  int repetitions = 0;
  do
  {
    e1 = this->calculateNeuralNetworkError();
    //lokale Variablen
    int numOfLearningEpochs = ivNumberOfLearningEpochs; 
    int numOfTrainingExamples = ivVNeuralNetTrainingExamples.size();
    float summedSquaredError  = 0.0,
          error               = 0.0;
  
    //Hauptschleife
    for (int n=0; n<numOfLearningEpochs; n++)
    {
      summedSquaredError = 0.0;
      for (int e=0; e<numOfTrainingExamples; e++)
      {
        float target = ivVNeuralNetTrainingExamples[e].second;
        this->setNeuralNetInputFromState
              ( ivVNeuralNetTrainingExamples[e].first );
        ivpNeuralNetwork->forward_pass( ivpNeuralNetwork->in_vec,
                                        ivpNeuralNetwork->out_vec );
        for (int o=0; o < ivpNeuralNetwork->topo_data.out_count; o++)
        {
          error = ivpNeuralNetwork->out_vec[o] - target;
          ivpNeuralNetwork->out_vec[o] = error;
          summedSquaredError += error * error;
        }
        ivpNeuralNetwork->backward_pass( ivpNeuralNetwork->out_vec, 
                                         ivpNeuralNetwork->in_vec);
      }
      ivpNeuralNetwork->update_weights();
    }
ivpNeuralNetwork->save_net("test.net");
    e2 = this->calculateNeuralNetworkError();
    ERROR_OUT<<"INFO: NN error after training: "<<e2<<endl;
    repetitions ++ ;
  }
  while (e2 >= initialError && repetitions < 5);

  ivpNeuralNetwork->save_net(ivNeuralNetworkFileName);
}

//############################################################################
// MAIN METHOD
//############################################################################

int main( int argc, char ** argv)
{
  EpisodeProcessingNeuralNetworkTrainer epnnt;
  
  epnnt.init( "epnnt.conf" );
  if (argc > 1)
  {
    epnnt.setEpisodeFileName( argv[1] );
  }
  
  epnnt.createTrainingData();
  epnnt.doTraining();
  
  return 0;
}

