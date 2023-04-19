# map_beautification
Dockerfile for map beautification using pix2ix

Build Command: 
'''
sudo docker build -t image-name . 
'''
Run Command: 
'''
sudo docker run -it --rm --gpus all image-name python p2p.py --edit "convert to minimalistic and clean floor plan" --input "test.jpg" --output "test_1.jpg" 
'''


Inputs: 
--edit : text prompt
--input : input file location inside docker
--output : output file location inside docker
--seed : random seed 
--step : no of iteration steps
--cfg-text : image weight
--cfg-image : text weight
