# Landmark-based Facial Expression Recognition

This folder contains an implemented demo of landmark-based facial expression recognition method provided by [OpenDR toolkit](https://opendr.eu).
The demo shows how the implemented *progressiv_spatio_temporal_bln_learner* can be used to recognize the expression from facial image sequence. 
However, the pretrained models is not provided publicly. 
In order to run the demo, you need to download one of the datasets, [AFEW](https://cs.anu.edu.au/few/AFEW.html), [CK+](https://www.pitt.edu/~emotion/ck-spread.htm), [Oulu-CASIA](https://www.oulu.fi/cmvs/node/41316), 
and the Dlib's landmark extractor from [here](http://dlib.net/face_landmark_detection.py.html) and train the PSTBLN model on the generated facial landmark dataset. 

This demo load a facial expression video as input, extracts the frames and then the facial landmarks, and employs a pretrained model to infer the facial expression. 


