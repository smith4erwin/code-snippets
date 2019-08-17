#!/bin/bash

time_format="+%Y-%m-%d %H:%M:%S"
tim=$(date "$time_format")
echo "tim: $tim"

echo current shell pid: $$

############################################################

script_dir=$(cd $(dirname $BASH_SOURCE) && pwd)

port=9191
bash ${script_dir}/start_imgsearch_srv_ol.sh monitor $port 1> ${script_dir}/log/monitor_${port}.log 2> ${script_dir}/monitor_${port}.log.err


