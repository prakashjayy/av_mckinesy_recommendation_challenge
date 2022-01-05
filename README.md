## Recommendation systems
[Source](https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-recommendation/)

A students solves a series of challenges. The problem here is to recommend the next 3 challenge given his past 10 challenges.

I have used Deep Learning to solve this problem.
Reason: That's where I exceed.


## Network Architecture:
- Input 10 challenges solved by the user. These 10 challenges are label encoded.
- A Embedding layer to represent each challenge using 50 Dim vector (Hyprer param. I used 50 to solve this problem)
- Concat all the vectors of the challenges.
- Pass them through a series of non-linear FC layers. ReLU is used for non-linearity.
- Final layer has 5501 neurons(num of challenges in the website).

## Training
- Used FocalLoss as the loss function as the number of positive labels is 3/5501 for each example. Default values of 0.25 and 2 were used for balancing param and focusing param.
- Initialized the final layer bias with math.log((1-0.01)/0.01) as given in FocalLoss paper. Remaining all initialization were default in Pytorch.
- Batchsize 256 and ran for 300 epochs. Network converged at 246 epochs.
- Training took approximately 3 hours ? Don't know the reason. Will check later.

## Results mAP@3
- Train mAP@3: 0.263 Val mAP@3: 0.1877
- Train Loss: 0.356 Val Loss: 0.4437

## Problems:
Due to lack of time didn't used challenge features provided with challenge dataset. Some important features which might work
1) Number of students who solved that challenge
2) Number of articles written by the author of that challenge
3) programming language used to solve this challenge

Thank you av for a very clean dataset.
