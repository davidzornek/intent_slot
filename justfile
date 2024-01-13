set dotenv-load
# set environmental variables here

build-image:
    #!/usr/bin/env bash
    sudo docker build -t nlp .
run-image:
    #1/usr/bin/env bash
    sudo docker run --gpus all -v ~/intent_slot/:/src/intent_slot/ -v ~/model_1_dataset/:/src -p 8888:8888 -it --name intent_slot nlp bash
