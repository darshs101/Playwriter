# Playwrite
Uses RNNs to read in lines from shakespeare plays and generate a new play.
This is done using a character predictive model that will take as input a variable length sequence and predict the next character. 
We can use the model many times in a row with the output from the last predicition as the input for the next call to generate a sequence.
