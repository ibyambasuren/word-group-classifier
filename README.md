# CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN

I will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and "hidden state" at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

Specifically, I'll train on a few thousand word from 16 group of origin, and predict which group a word is from based on the spelling.
