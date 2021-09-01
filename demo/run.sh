
source ~/torch-env/bin/activate

cd ~/lh/projects/opendr_internal/projects/perception/activity_recognition/demos/online_recognition

pwd

git checkout au-demo-har

PYTHONPATH=~/lh/projects/opendr_internal/src python3 demo.py --ip 0.0.0.0 --port 8000 --algorithm x3d --model m --device=cuda

PYTHONPATH=~/lh/projects/opendr_internal/src python3 demo.py --ip 0.0.0.0 --port 8000 --algorithm cox3d --model m --device=cuda

