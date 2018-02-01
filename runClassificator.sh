#!/usr/bin/env bash
spadeAddress=$(ping spade -c 1 | grep "172\." | awk -F "(" 'NR==1{print $2}' | awk -F ")" '{print $1}')
echo "SPADE_ADDRESS: $spadeAddress"
sleep 5
echo "wait for 5"
#sleep 5
./agentClassificator.py "$spadeAddress" #"$spadeAddress"
