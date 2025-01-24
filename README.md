# Magic the Gathering Cube Multi-Purpose Model

## Overview

This repo has three components:
- The model
- The demo server
- The demo client

## Setup 

To train, you'll need to download data. I made processed data publicly available here: https://drive.google.com/drive/folders/1Pzsh_uZdydFKLxzwIYZfdq4d2TRbjpSn?usp=share_link

Set it up like this:
```
/data
    /train
        ...files
    /test
        ...files
```

A trained model is comitted to the repo under `/model/tfjs_model`. You can use this model to run the demo.

## The Model

You'll need python installed with tensorflow.

Navigate to `/model/`

You have a couple executable files there:

- `python train.py 10 128 false 1.0` trains the model with:
    - 10 epochs
    - 128 batch size
    - false continuing training from a previous model
    - 1.0 for the loss weights
- `python test.py` tests the model

As well as a conversion script. This script needs to be run from the root folder like:
`sh model/convert.sh`

In order to get the conversion script to work, you'll need to install tfjs binaries which are not supported in windows currently.

## Demo

First run the server from `/demo/server` with `npm run start`

Then run the client from `/demo/client` with `npm start`

Wait a minute, and then a browser window should open with the demo at `localhost:3000`

## Uploading data to s3

To recursively upload the `data` folder to s3, run
`aws s3 cp data s3://cubecobra-private/training-2025/data --recursive`

And to download it, run
`aws s3 cp s3://cubecobra-private/training-2025/data data --recursive`

To upload the tfjs model to s3, run
`aws s3 cp model/tfjs_model s3://cubecobra-data-production/model --recursive`
MAKE SURE to get the indexToOracleMap file as well, or the model won't work.


