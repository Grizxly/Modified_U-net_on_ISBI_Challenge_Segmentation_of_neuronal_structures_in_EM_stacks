# Modified_U-net_on_ISBI_Challenge_Segmentation_of_neuronal_structures_in_EM_stacks
Implement a modified U-net, train and run for segmentation tasks on neuronal structures in EM stack
* This is the work of COMP9517 project of UNSW, 2019
* Dataset can be download there: http://brainiac2.mit.edu/isbi_challenge/home
* For more detail about the model please refer to 'report.pdf'
* The model runs on Tensorflow 1.14 ONLY

To use the model:
1. Download the original dataset from the source mentioned above.
2. Run preprocessing.py to generate augmented training set
3. Run train.py to train the model 
4. Run run.py to run trained model on desired input

cheers
