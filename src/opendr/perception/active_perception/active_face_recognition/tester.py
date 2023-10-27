from opendr.perception.active_perception.active_face_recognition import ActiveFaceRecognitionLearner

# learner = ActiveFaceRecognitionLearner(iters=500, n_steps=100)
# learner.fit()
# learner.save('./best')
# print('saved')
# del learner
learner = ActiveFaceRecognitionLearner()
learner.load('./best.zip')
print('loaded')
obs = learner.env.reset()
while True:
    print('evaluating')
    action, _ = learner.infer(obs)
    obs, reward, done, info = learner.env.step(action)
    if done:
        obs = learner.env.reset()

