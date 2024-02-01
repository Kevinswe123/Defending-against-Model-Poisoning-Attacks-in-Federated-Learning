#!/usr/bin/env bash

GPU=$1

DATASET=$2

NET=$3

BIAS=$4

NWORKERS=$5

BATCH_SIZE=$6

LR=$7

NITER=$8

SERVER_PC=$9

SEED=$10

NBYZ=$11

BYZ_TYPE=$12

AGGREGATION=$13

P=$14

TRIGGERLABEL=$15

PORTION=$16

CREATNEWWATERMARKDATA=$17

THRESHOLD=$18

DBDATASET=$19
#TV1=$15



python3 ./test_byz_p1.py \
--gpu $GPU \
--dataset $DATASET \
--net $NET \
--bias $BIAS \
--nworkers $NWORKERS \
--batch_size $BATCH_SIZE \
--lr $LR \
--niter $NITER \
--server_pc $SERVER_PC \
--nrepeats $SEED \
--nbyz $NBYZ \
--byz_type $BYZ_TYPE \
--aggregation $AGGREGATION \
--p $P \
--trigger_label $TRIGGERLABEL \
--poisoned_portion $PORTION \
--creatNewWatermarkDataset $CREATNEWWATERMARKDATA \
--threshold $THRESHOLD \
--dbdataset $DBDATASET \
#--trust_value1 $TV1 \


