#!/bin/bash

time_format="+%Y-%m-%d %H:%M:%S"
time_format1="+%y%m%d-%H%M%S"

tim=$(date "$time_format")
echo "tim: $tim"

echo current shell pid: $$

BASH_NAME=start_imgsearch_srv_ol.sh
############################################################

workdir=$(cd $(dirname $BASH_SOURCE)/.. && pwd)
source $workdir/script/config.sh
export PATH=$workdir/venv/bin:$PATH
echo $(which python)

srcdir=$workdir/src
appfile=$srcdir/app.py

port=$2
echo port:$port

piddir=$deploy_dir/pid
pidfile=$piddir/cbir-${port}.pid

logdir=$deploy_dir/log
logfile=$logdir/cbir-${port}.log


#exit

warn()
{
   echo ">>>>> warn"
   msg="$1 at $(date "$time_format") on $(hostname)"
   echo "msg:  $msg"
   python $srcdir/bin/alert.py "$msg"
#  $PREFIX_MYPYTHON $MYPYTHON $srcdir/bin/alert.py "$msg"
}


#checkAlive appfile pidfile port
checkAlive()
{
    echo ">>>>> checkAlive"
    run=$(ps -ef | grep $1 | grep $3 | grep -v "grep")
    echo "check alive results: "$run
    if [ -n "$run" ]; then
        PID=$(echo $run | tr -s ' ' | cut -d ' ' -f 2)
        echo $PID > $2
        return 0
    else
        return 1
    fi
}


#cmdStart appfile pidfile logfile port
cmdStart()
{
    echo ">>>>> cmdStart"
    checkAlive $1 $2 $4
    if [ $? = 0 ]; then
        echo $(date "$time_format")" Process has been started, "$1,$4,$PID
        return 0
    else
        local logbak
        logbak=.$(date "$time_format1").bak
        mv $3 $3$logbak
        nohup python $1 -p $4 &> $3 &
#       nohup $PREFIX_MYPYTHON $MYPYTHON $1 -p $4 &> $3 &
        sleep 50    #need time to load data, may be failed, so wait a little longer
        
        checkAlive $1 $2 $4
        if [ $? = 0 ]; then
            echo $(date "$time_format")" Process start, "$1,$4,$PID
            return 0
        else
            echo $(date "$time_format")" Process failed, "$1,$4
            mv $3$logbak $3
            rm $2
            return 1
        fi
    fi
}


#cmdStop appfile pidfile port
cmdStop()
{
    echo ">>>>> cmdStop"
    checkAlive $1 $2 $3
    if [ $? = 0 ]; then
        kill -9 $PID
        echo "KILL -9 $PID"
        rm $2
        echo $(date "$time_format")", Process is killed, "$1,$3,$PID
        return 0
    else
        echo $(date "$time_format")", Process may not be start, "$1,$3
        return 1
    fi
}


#cmdRestart appfile pidfile logfile port
cmdRestart()
{
    echo ">>>>> cmdRestart"
    cmdStop $1 $2 $4
    cmdStart $1 $2 $3 $4
}


#monitorBody appfile pidfile logfile port
monitorBody()
{
    echo ">>>>> monitorBody"
    checkAlive $1 $2 $4
    if [ $? = 0 ]; then
        echo $(date "$time_format")' first check alive: true'
    else
        warn "$1,$4 is not alive at first check, rechecking!!!"
        echo $(date "$time_format")' first check alive: false'
        sleep 20

        checkAlive $1 $2 $4
        if [ $? = 0 ]; then
            echo $(date "$time_format")' second check alive: true'
        else
            warn "$1,$4 is not alive after checking twice, restarting!!!"
            echo $(date "$time_format")' second check alive: false'
            cmdRestart $1 $2 $3 $4
        fi
    fi
}


#cmdMonitor $appfile $pidfile $logfile $port
cmdMonitor()
{
    echo ">>>>> cmdMonitor"
    cmdStart $1 $2 $3 $4
    sleep 10
    while : ; do
        monitorBody $1 $2 $3 $4
        sleep 40
    done
}


case $1 in
    monitor)
        cmdMonitor $appfile $pidfile $logfile $port
        ;;
    start)
        cmdStart   $appfile $pidfile $logfile $port
        ;;
    stop)
        cmdStop    $appfile $pidfile $port
        cmd_monitor=$(ps -ef | grep monitor | grep $BASH_NAME | grep $port | grep -v "grep") #只针对单个进程，如果启动多个，无法kill
        echo cmd_monitor:$cmd_monitor
        if [ -n "$cmd_monitor" ]; then
            MONITOR_PID=$(echo $cmd_monitor | tr -s ' ' | cut -d ' ' -f 2)
            echo MONITOR_PID:$MONITOR_PID
            kill -9 $MONITOR_PID
        fi
        ;;
    restart)
        cmdRestart $appfile $pidfile $logfile $port
        ;;
    status):
        checkAlive $appfile $pidfile $port
        if [ $? = 0 ]; then
            exit 0
        else
            exit 1
        fi
        ;;
    *)
        echo 'wrong parameter'
        ;;
esac

