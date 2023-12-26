# Pikachu Volleyball AI
This is the final project of Introduction to Machine Learning. We try to use reinforce learning to get the better player of the game "pikachu volleyball".
![image](https://github.com/torrid-fish/Pikachu_Volleyball_AI/assets/92087101/92517d2b-f18b-4faa-b28f-5883f41a69f9)

## Report
Here is our report:
- [Team6_Final_Report.pdf](https://github.com/torrid-fish/Pikachu_Volleyball_AI/files/13771154/Team6_Final_Report.pdf)

## Result
We’ve tried several layers of neural network and compared their performance. 
First of all, we use two layers and couldn’t get a satisfying result after a long time. We guessed that the number of the node might not enough to deal with all the situation so that the win rate could not go higher. 
Hence, we add more number to the second layer, and the win rate improved to 37%. After times of trying, we found that the three layers could meet our requirement with acceptable memory size. 
Below is the comparison of 3 different models we trained.
![image](https://github.com/torrid-fish/Pikachu_Volleyball_AI/assets/92087101/5c42fb96-27c0-4093-9013-499a3d49b3fb)

We also compare the result between whether using PER (Priortized Experience Replay):
![image](https://github.com/torrid-fish/Pikachu_Volleyball_AI/assets/92087101/1d05f1cb-bfa1-440f-bed8-ae3b7e7e397c)

Finally, we also try to use APEX to enhence the performance:
![image](https://github.com/torrid-fish/Pikachu_Volleyball_AI/assets/92087101/a6fdebb4-b55a-4ffb-8ddc-eb86972c526e)


