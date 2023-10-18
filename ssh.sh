#!/bin/bash
IP=$1
PORT=$2
service ssh start
ssh -p 2222 -L 8080:$IP:$PORT $IP
