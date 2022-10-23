# tech-talk-01
Berlin Tech-Talk to deploy a GPT3 and Whisper Service

## Run Service locally
01. Change server_name="0.0.0.0" to server_name="127.0.0.1"

## Install microk8s with helm
01. cd meetup-helm-chart
02. sudo ./deploy_with_microk8s.sh

## Build Image for microk8s locally
01. sudo docker build -t localhost:32000/gpt:1.1.1 .
02. sudo docker push localhost:32000/gpt:1.1.1