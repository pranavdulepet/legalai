## Overview

This repository contains code for a Document Management System.

Features:
* Document Classification
* Document Similarity
* Document Search
* Keyphrase Extraction
* Named Entity Recognition
* Extractive Summarization

## Run

### Prerequisites
Install docker, docker-compose, and aws cli on your machine.
Setup aws cli credentials with admin priviledges.

### Local Development

During development on a single node:

#### /etc/hosts
Set this in /etc/hosts file in your pc
> mydomain.tld        127.0.0.1

> api.mydomain.tld    127.0.0.1

> mydomain.tld/whoami 127.0.0.1

Do so for all subdomains in your app

#### Env Variables
Set this environmental variable in your node:

> export DOMAIN=mydomain.tld

Set repo name env variable for traefik network labels:

> export REPO_NAME=dms_repo

#### CMD
Then run:
> docker-compose up

Or, to be more specific:
> docker-compose -f docker-compose.yml -f docker-compose.override.yml up

### Build
Once the development and testing is complete, you can build and tag the api with:
> docker build -f Dockerfile -t 709721532782.dkr.ecr.us-west-1.amazonaws.com/project_dms:<TAG> -t 709721532782.dkr.ecr.us-west-1.amazonaws.com/project_dms:latest
  
Authenticate aws to docker hub:
> aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 709721532782.dkr.ecr.us-west-1.amazonaws.com

Push the image to AWS ECR:
> docker push 709721532782.dkr.ecr.us-west-1.amazonaws.com/project_dms:<TAG>

### Deployment

To deploy using docker swarm on multiple nodes, copy the `docker-compose.yml` and `app/.config` files to the master node and run:
> docker swarm init

> docker network create --driver=overlay traefik-public

> docker network create --driver=overlay backend

To add more nodes to this swarm run in your master node:
> docker swarm join-token worker 

Copy and paste the command in the output in the worker node.

You may want to add some placement contraint labels to your nodes and the compose files.
> export NODE_ID=$(docker info -f '{{.Swarm.NodeID}}') # or whichever node you want to add the labels to

> docker node update --label-add some-label-name=true $NODE_ID

Add this segment in your compose file under any service deploy:
```yaml
  deploy:
      placement:
        constraints:
          - node.role == manager # or node.role == worker
          - node.labels.some-label-name == true
```        

Copy `app/.config` file to your node in the same directory as docker-compose.yml.

Then edit the values in `.config` as per your requirement.

Set this environmental variable in your node:
> export DOMAIN=mydomain.tld

Do this in every node where `api` service needs to be replicated:

Authenticate aws to docker hub:
> aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 709721532782.dkr.ecr.us-west-1.amazonaws.com

Pull the image from AWS ECR:
> docker pull 709721532782.dkr.ecr.us-west-1.amazonaws.com/project_dms:<TAG>
  
Then run:
> docker stack deploy -c docker-compose.yml name-of-stack

## Troubleshoot

### Elasticsearch
Sometimes you need to set this in your instance for Elasticsearch container to work:
> sysctl -w vm.max_map_count=262144

If there is insufficient storage in your PC do:

> curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{
> 	"transient": {
> 		"cluster.routing.allocation.disk.watermark.low": "93%",
> 		"cluster.routing.allocation.disk.watermark.high": "94%",
> 		"cluster.routing.allocation.disk.watermark.flood_stage": "95%"
> 	}
> }'

### Docker
If docker says permission denied, try:

> sudo chmod 666 /var/run/docker.sock

and/or 

> delete ~/.docker/config.json




