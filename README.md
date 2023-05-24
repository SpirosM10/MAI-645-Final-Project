# MAI-645-Final-Project
MAI 645 Final Project
-How to run this code-

1. Generating the train data:

To Generate the training data, download BVH files from
the motion capture database and store in a file on your computer.
Run the generate_train_data.py depending on what representation
you want to test out. This will create a new folder in the
directory that is input to the generate function.

2. Training the model:

To train the model, run the pytorch_train_aclstm_euler.py file,
making sure to change the directories of the files based on what
representation you wish to train. To improve the training
procedure, change the loss functions to the appropriate
ones for each representation.

3. Testing the model:

To test the model, run the pytorch_test_synthesize_motion, 
making sure to change the "for_quantitative" vartiable to
false or true depending on the purpose of the test. When set to True,
the model will only use the 01.bvh file as a initial 
sequence which then is used to generate the rest of the motion.
Make sure to change all directories to match the representation 
you wish to evaluate, keeping in mind to change the function that 
writes back to the bvh format. This bvh file is then generated into 
training data using the positional representation and  compared
with the groundtruth to find the loss between the two. For the qualitative evaluation,
please load the bvh files found in the folders ending with /...representation_qualitative
using this online bvh player http://lo-th.github.io/olympe/BVH_player.html
