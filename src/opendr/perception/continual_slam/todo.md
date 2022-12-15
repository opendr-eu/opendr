- [ ] Decide on software architechture 
    * [ ] Decide on which classes are going to be used from CL-SLAM repo
    * [ ] Decide how they are going to be implemented
    * [ ] Since no loop closure will be added for now, decide on 
    if we need to install g2opy
    * [ ] Datasets ?
    * [ ] Update frequency ?
        * [ ] Assuming that we have two nodes, namely, constant Publisher and Learner?, we might need to obtain forward pass time for publisher and total adapt time for learner, in order to decide on how frequent we are going to update the parameters of publisher. 
        * [ ] Further, we can also try to write a simple async interrupt routine to basically update the parameters as they come at anytime without caring about the frequency
- [ ] Finalize the architechture
- [ ] Write ROS nodes
- [ ] Ask about where we will deploy? under perception?
- [ ] Create PR and delete this file