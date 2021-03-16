#!/bin/bash

cd $(dirname $0)
BASEDIR=`pwd`

SCRIPTDIR=scripts
AGENTDIR=agent/bin
COACHDIR=coach/bin

AGENT=FRA-UNIted_Agent
COACH=FRA-UNIted_Coach

HOST=localhost

SLEEPTIME=0.5
SLEEPTIME_GDB=0.5

AGENTOPT="-log_lev 4 -log_dir /tmp -log_deb 001 -log_def 001 -log_wm 001 -log_mdp 000 -log_err 101 -host $HOST"
#AGENTOPT="-wm_test 1 -log_lev 4 -log_dir /tmp -log_deb 001 -log_def 001 -log_wm 001 -log_mdp 000 -log_err 101 -host $HOST"
#AGENTOPT="-wm_test 1 -log_lev 0 -log_dir /tmp -log_deb 000 -log_def 000 -log_wm 000 -log_mdp 000 -log_err 000 -host $HOST"
#AGENTOPT="-host $HOST"
COACHOPT="-server_9.4 0 -host $HOST"

USERAMLOGGING=false
AGENTOPTWITHRAMLOGGING="-log_lev 4 -log_dir `cat $BASEDIR/scripts/ramDiskPath` -log_deb 001 -log_def 001 -log_wm 001 -log_mdp 000 -log_err 101 -host $HOST"
COACHOPTWITHRAMLOGGING="-server_9.4 0 -host $HOST -log_dir `cat $BASEDIR/scripts/ramDiskPath`"

####### Test wether RamDisk logging schould be used or not ########
cd $SCRIPTDIR
#echo `cat ./scripts/useRamDisk`
if [ "`cat ./useRamDisk`" = "true" ]
then
	./isRamDiskNowActive
	if [ $? -eq 0 ]
	then
		USERAMLOGGING=true
	else
		echo "Need superuser rights to initialize RamDisk for logging into RAM"
		./initRamDisk
		./isRamDiskNowActive
		if [ $? -eq 0 ]
		then
			USERAMLOGGING=true
		else
			echo "RamDisk initialization failed!"
		fi
	fi
fi

if [ $USERAMLOGGING = true ]
then
	echo "RamLogging is in use!"
	AGENTOPT=$AGENTOPTWITHRAMLOGGING
	COACHOPT=$COACHOPTWITHRAMLOGGING
fi

cd $BASEDIR

# sleep in between agent starting if available
SLEEP=`which sleep`

if [ "$SLEEP" != "" ]; then
  SLEEP="$SLEEP $SLEEPTIME"
else
  echo "WARNING: No sleep found on this system!"
fi

######## START AGENTS ###########

cd $AGENTDIR || exit 1

#I=0
#while [ "$I" != "11" ] ; do
#  echo $I ter LAUF! BEGIN!
#  if [ "$I" == "0" ] ; then
#    if [ "$1" != "debugA1" ] ; then
#      ./$AGENT -goalie $AGENTOPT &
#    else
#      ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
#      gdb -ex "file ./$AGENT" -ex "set args -goalie $AGENTOPT" -ex "break $2" -ex "run"
#      echo SUCCESS! IN BACKGROUND!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    fi
#  else
#    if [ "$1" != "debugA"`expr $I + 1` ] ; then
#      ./$AGENT $AGENTOPT &
#    else
##      gdb ./$AGENT $AGENTOPT &
#      ./$AGENT $AGENTOPT &
#    fi
#  fi
#  $SLEEP
#
#  echo $I ter Lauf!
#
#  I=`expr $I + 1`
#done
#
#echo AFTER AGENTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if [ "$1" != "debugA1" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT -goalie $AGENTOPT &
  else
    ./$AGENT -goalie $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args -goalie $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA2" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA3" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA4" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA5" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA6" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA7" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA8" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA9" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA10" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP

if [ "$1" != "debugA11" ] ; then
  if [ "${1:0:5}" != "debug" ] ; then
    ./$AGENT $AGENTOPT &
  else
    ./$AGENT $AGENTOPT > /dev/null 2>&1 &
  fi
else
  ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
  gdb -q -ex "file ./$AGENT" -ex "set args $AGENTOPT" -ex "break $2" -ex "run"
fi

$SLEEP


######## START COACH ##########
if [ "$1" != "woc" ] ; then

  echo

  cd $BASEDIR
  cd $COACHDIR || exit 1

  if [ "$1" != "debugC" ] ; then
    if [ "${1:0:5}" != "debug" ] ; then
      ./$COACH $COACHOPT &
    else
      ./$COACH $COACHOPT > /dev/null 2>&1 &
    fi
  else
    ${BASEDIR}/${SCRIPTDIR}/waitUntilGdbIsRdyToBeSentInBg $SLEEPTIME_GDB &
    gdb -q -ex "file ./$COACH" -ex "set args $COACHOPT" -ex "break $2" -ex "run"
  fi

fi

cd $BASEDIR

echo

if [ "${1:0:5}" == "debug" ] ; then
  fg
fi
