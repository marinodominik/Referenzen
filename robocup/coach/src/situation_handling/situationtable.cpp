#include "situationtable.h"
#include <fstream>
#include <sstream>
#include "str2val.h"
#include "defs.h"
#include "param.h"
#include "logger.h"

using namespace std;

//###########################################################################
// KLASSE SituationTable
//
//
//###########################################################################


//===========================================================================
// ?FFENTLICHE METHODEN (sowie Freunde)
//===========================================================================

//---------------------------------------------------------------------------
// DEFAULT-KONSTRUKTOR
//---------------------------------------------------------------------------
SituationTable::SituationTable()
  { 
    ivNumRows= 0;
    ivNumCols= 0;
    ivMaxRows=0;
    ivppTableArray = NULL;
  }

//---------------------------------------------------------------------------
// DESSTRUKTOR
//---------------------------------------------------------------------------
SituationTable::~SituationTable()
  {
  }

//---------------------------------------------------------------------------
// METHOD getNumberOfColumns()
//---------------------------------------------------------------------------
int
SituationTable::getNumberOfColumns()
{
	return ivNumCols;
}

//---------------------------------------------------------------------------
// METHOD createRandomSituationTable
//---------------------------------------------------------------------------
bool 
SituationTable::createRandomSituationTable(int rows, bool ball, int numPlayers) 
{
  int columnCounter = 0;
  
  ivNumRows = 0;
  ivNumCols = 0;
  if (ball) ivNumCols += 4;
  ivNumCols += 5 * numPlayers;
  if (rows > 50)
    this->setMaxRows(rows);
  else
    this->setMaxRows(50);

  float *tmp= 0;

  for (int i=0; i<rows; i++)
  {
    if (tmp==0) 
      tmp = new float[ivNumCols];
  	//BALL
  	if (ball)
    {
	  tmp[0] = this->getRandomBallPosX();
	  tmp[1] = this->getRandomBallPosY();
	  bool velOk = false;
	  while (!velOk)
	  {
  	    tmp[2] = this->getRandomBallVelX();
	    tmp[3] = this->getRandomBallVelY();
	    float ballSpeedDivisor = 1.0;
	    velOk = this->checkBallVelocity(tmp[2], tmp[3], ballSpeedDivisor);
	  }
	  columnCounter = 4;
    }
  	//SPIELER
  	for (int j=0; j<(5*numPlayers); j++)
  	{
  	  if (j%5 == 0) tmp[ j + columnCounter ] = this->getRandomPlayerPosX();
  	  if (j%5 == 1) tmp[ j + columnCounter ] = this->getRandomPlayerPosY();
  	  if (j%5 == 2) tmp[ j + columnCounter ] = this->getRandomPlayerAngle();
  	  if (j%5 == 3) 
  	  {
   	  	bool velOk = false;
   	  	while (!velOk)
   	  	{
   	      tmp[ j + columnCounter    ] = this->getRandomPlayerVelX();
   	      tmp[ j + columnCounter + 1] = this->getRandomPlayerVelY();
   	      float ballSpeedDivisor = 1.0;
   	      velOk = this->checkBallVelocity
   	              (tmp[j + columnCounter], tmp[j + columnCounter + 1],
   	               ballSpeedDivisor);
   	  	}
	  }
  	}

    ivppTableArray[ivNumRows]= tmp;
    ivNumRows++;
    tmp= 0;
  }

  if (tmp)
    delete[] tmp;
  return true;
}

//---------------------------------------------------------------------------
// METHOD setOffObject
//---------------------------------------------------------------------------
void 
SituationTable::setOffObject(int index)
{
	for (int i=0; i<ivNumRows; i++)
	{
    float *certainRow = ivppTableArray[i];
    if (certainRow == NULL)
    {
      cout<<"ERROR: In ivppTableArray["<<i<<"] I found NULL."<<endl;
      return;
    } 
		if ( index==0 ) //Ball im Nirvana platzieren
		{
			certainRow[0] = -1000;
			certainRow[1] = -1000;
			certainRow[2] = 0;
			certainRow[3] = 0;
		}
		else
		{   //Spieler im Nirvana platzieren
			certainRow[ (index-1)*5 + 4 + 0 ] = -1000;
			certainRow[ (index-1)*5 + 4 + 1 ] = -1000;
			certainRow[ (index-1)*5 + 4 + 2 ] = 0;
			certainRow[ (index-1)*5 + 4 + 3 ] = 0;
			certainRow[ (index-1)*5 + 4 + 4 ] = 0;
		}
	}
}

//---------------------------------------------------------------------------
// OPERATOR << for SituationTable
//---------------------------------------------------------------------------
ostream& 
operator<< (ostream& o,const SituationTable& t) 
{
  cout << "[";
  cout.precision(5);
  for (int st= 0; st < t.ivNumRows; st++) 
  {
    cout << endl << setw(3) << st << ":";
    for (int act= 0; act < t.ivNumCols; act++) 
    {
      if (act%10==0) cout << endl<<flush;
      cout << setw(8) << t.getElement(st,act);
    }
  }
  cout << "\n]"<<flush;
  return o;
}

//---------------------------------------------------------------------------
// METHOD setMaxRows
//---------------------------------------------------------------------------
void 
SituationTable::setMaxRows(int max) 
{
//  if (ivMaxRows <> max)
//    return;
  ivMaxRows= max;
  float ** tmp= new float*[ivMaxRows];
  for (int row=0; row<ivNumRows; row++)
    tmp[row]= ivppTableArray[row];
  for (int row=ivNumRows; row < ivMaxRows; row++) 
    tmp[row]= 0;
  if (ivppTableArray) delete[] ivppTableArray;
  ivppTableArray= tmp;
}

//---------------------------------------------------------------------------
// METHOD getElement() 
//---------------------------------------------------------------------------
float 
SituationTable::getElement(int row,int col) 
const
{ 
  if (    col < 0 
       || col >= ivNumCols 
       || row < 0 
       || row >= ivNumRows) 
    cerr << "\n(" << row <<"," << col << ") not in range"<<flush;
  return ivppTableArray[row][col]; 
}

//---------------------------------------------------------------------------
// METHOD getRow() 
//---------------------------------------------------------------------------
float * 
SituationTable::getRow(int row) 
const
{
  if (row < 0 || row >= ivNumRows)  
    cerr << "\n(" << row << ") not in range"<<flush;
  return ivppTableArray[row];
}

//---------------------------------------------------------------------------
// METHOD set 
//---------------------------------------------------------------------------
void 
SituationTable::set(int row, int col, float value) 
{
  ivppTableArray[row][col]= value;
}

//---------------------------------------------------------------------------
// METHOD save 
//---------------------------------------------------------------------------
bool 
SituationTable::save(const char* fname ) 
const 
{
  ofstream out(fname);

  for (int row= 0; row < ivNumRows; row++) 
  { 
    LOG_FLD(1, "SituationTable: Save row "<<row
      <<" of situation table to file "<<fname);
    out << "\n"<<flush;
    for (int col= 0; col < ivNumCols; col++) 
      out << " " <<  setw(6) << ivppTableArray[row][col];
  }
  out<<flush;
  out.close();
  return true;
}

//---------------------------------------------------------------------------
// METHOD load
//---------------------------------------------------------------------------
bool 
SituationTable::load(int cols,const char* fname ) 
{
  const int MAX_LINE_LEN= 1024;
  char line[MAX_LINE_LEN];
  
  ivNumRows= 0;
  ivNumCols= cols;
  this->setMaxRows(50);

  FILE *infile=fopen(fname,"r");
  if (infile==NULL)
  {
    fprintf(stderr,"File %s can't be opened\n", fname);
    return false;
  }

  float *tmp= 0;

  while( fgets(line,MAX_LINE_LEN,infile) != NULL)
  {
    if(*line=='\n'||*line=='#')
      continue;  /* skip comments */

    for (int i=0;i <MAX_LINE_LEN; i++) 
      if (line[i]== '\n') 
      {
    	line[i]= '\0';
	    break;
      }

    if (tmp==0) 
      tmp= new float[ivNumCols];

    bool warning= false;
    int res= str2val( line, ivNumCols, tmp);

    if (res == ivNumCols) 
    {
      if (ivNumRows >= ivMaxRows)
      {
	    this->setMaxRows(ivMaxRows+50);
      }
//cout<<"LOAD -  4g"<<endl<<flush;
//cout<<"        ivNumRows = "<<ivNumRows<<endl<<flush;
//cout<<"        tmp       = "<<tmp<<endl<<flush;
//cout<<"        ivppTableArray[ivNumRows] = "<<flush<<ivppTableArray[ivNumRows]<<endl<<flush;
      ivppTableArray[ivNumRows]= tmp;
      ivNumRows++;
      tmp= 0;
    }

    if (res != ivNumCols || warning) 
    {
      cout << "\n problems with reading line = " << line<<flush;
      cout << "res= " << res << ", warning = " << warning<<flush;
    }
  }
  if (tmp)
    delete[] tmp;
  fclose(infile);  
  return true;
}

//---------------------------------------------------------------------------
// METHOD toString
// CONVERSION of the whole TABLE to a STRING
//---------------------------------------------------------------------------
string SituationTable::toString(int num)
{
  ostringstream ost;
  for (int i=0; i < ivNumCols; i++){
    if(i>0) ost << " ";
    ost << ivppTableArray[num][i]<<flush;
  }
  return ost.str();
}

//===========================================================================
// PRIVATE METHODEN (Hilfsfunktionen)
//===========================================================================

//---------------------------------------------------------------------------
// METHOD getRandomPlayerPosWithinRange
//---------------------------------------------------------------------------
float 
SituationTable::getRandomPlayerPosWithinRange(float from, float to)
{
  	float r = TG_MY_RANDOM_NUMBER;
    r = from + (r*(to-from));	
    return r;
}

//---------------------------------------------------------------------------
// METHOD getRandomPlayerPosX
//---------------------------------------------------------------------------
float
SituationTable::getRandomPlayerPosX()
{
	float r = TG_MY_RANDOM_NUMBER;
	r = (2*r) - 1;
	return FIELD_BORDER_X * r;
}
//---------------------------------------------------------------------------
// METHOD getRandomPlayerPosY
//---------------------------------------------------------------------------
float
SituationTable::getRandomPlayerPosY()
{
	float r = TG_MY_RANDOM_NUMBER;
	r = (2*r) - 1;
	return FIELD_BORDER_Y * r;
}
//---------------------------------------------------------------------------
// METHOD getRandomPlayerAngle
//---------------------------------------------------------------------------
float 
SituationTable::getRandomPlayerAngle()
{
  return (2*PI) * TG_MY_RANDOM_NUMBER;
}
//---------------------------------------------------------------------------
// METHOD getRandomPlayerVelX
//---------------------------------------------------------------------------
float
SituationTable::getRandomPlayerVelX()
{
	return (2.0 * ServerParam::player_speed_max * TG_MY_RANDOM_NUMBER)
	       - ServerParam::player_speed_max;
}
//---------------------------------------------------------------------------
// METHOD getRandomPlayerVelY
//---------------------------------------------------------------------------
float
SituationTable::getRandomPlayerVelY()
{
  return this->getRandomPlayerVelX();
}
//---------------------------------------------------------------------------
// METHOD getRandomBallPosX
//---------------------------------------------------------------------------
float
SituationTable::getRandomBallPosX()
{
	return this->getRandomPlayerPosX();
}
//---------------------------------------------------------------------------
// METHOD getRandomBallPosY
//---------------------------------------------------------------------------
float
SituationTable::getRandomBallPosY()
{
	return this->getRandomPlayerPosY();
}
//---------------------------------------------------------------------------
// METHOD getRandomBallVelX
//---------------------------------------------------------------------------
float
SituationTable::getRandomBallVelX()
{
	float min = - ServerParam::ball_speed_max;
	float delta = 2 * ServerParam::ball_speed_max;
	return min + (delta * TG_MY_RANDOM_NUMBER);
}
//---------------------------------------------------------------------------
// METHOD getRandomBallVelY
//---------------------------------------------------------------------------
float
SituationTable::getRandomBallVelY()
{
  return this->getRandomBallVelX();
}
//---------------------------------------------------------------------------
// METHOD checkBallVelocity
//---------------------------------------------------------------------------
bool 
SituationTable::checkBallVelocity(float vx, float vy, float ballSpeedDivisor)
{
	if (sqrt(vx*vx + vy*vy) >= ServerParam::ball_speed_max / ballSpeedDivisor)
	{
	  cout<<"PROBLEM: "<<sqrt(vx*vx + vy*vy)<<" >= "<<ServerParam::ball_speed_max / ballSpeedDivisor<<endl;
	  return false;
	}
	return true;
}
//---------------------------------------------------------------------------
// METHOD checkPlayerVelocity
//---------------------------------------------------------------------------
bool 
SituationTable::checkPlayerVelocity(float vx, float vy)
{
	if (sqrt(vx*vx + vy*vy) >= ServerParam::player_speed_max)
	  return false;
	return true;
}
