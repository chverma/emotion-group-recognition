#!/usr/bin/env bash
export spadeAddress=$(ping spade -c 1 | grep "172\." | awk -F "(" 'NR==1{print $2}' | awk -F ")" '{print $1}')
export webServerAddress=$(ping web_server -c 1 | grep "172\." | awk -F "(" 'NR==1{print $2}' | awk -F ")" '{print $1}')
echo "SPADE_ADDRESS: $spadeAddress"
sleep 5
echo "wait for 5"
#sleep 5
./agentCoordinator.py "$spadeAddress" #"$spadeAddress"
