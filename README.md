# Magic the Gathering Cube Multi-Purpose Model

This repo contains a model that simultaneously trains a deckbuiler, drafter, and recommender system with a shared encoder.

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

A trained model is comitted to the repo under `tfjs_model`. You can use this model to run the demo.

## Training a New Model

```bash
>>> curl -LsSf https://astral.sh/uv/install.sh | sh # install uv, which we use for dependency management
>>> uv sync
>>> source .venv/bin/activate
>>> python ccml/train.py 500 256 false cube
```

The above command will train a model with a batch size of 256 such that training will stop after it has iterated over each cube 500 times. That last parameter can be any of cube, card, pick, deck. Note that if you select a number that would not let it train on the full dataset, the script will train longer until it goes over each data point at least one time.

Once your model is done training (or you manually kill the process), your model will be saved in the `model` directory. Execute the following script to convert the model to js in order to deploy it or launch the demo. In order to get the conversion script to work, you'll need to install tfjs binaries which are not supported in windows currently.

```bash
>>> sh scripts/convert.sh
```

Additionally, to get some statistics on the performance of the holdout set, you can run

```bash
>>> python ccml/test.py
```

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
`aws s3 cp tfjs_model s3://cubecobra-data-production/model --recursive`
MAKE SURE to get the indexToOracleMap file as well, or the model won't work.


