import torch
import equal_diff_set

# We probably want to vary some of these parameters
num_properties = 4
num_values = 3
# This is the number of potential sets evaluated by an agent in a sequence
length_sequence = 30
num_sequences_train = 10000
num_sequences_test = 500

# Experiment 1 is making the training data have faulty agents
# Then test using perfect agent
# There are two variants:
game = equal_diff_set.set_game(num_properties, num_values)
perfect_mask = torch.ones((1, num_properties))
random_single_fault_masks = torch.ones((num_properties, num_properties))
random_single_fault_masks_with_perfect = torch.ones((num_properties+1, num_properties))
for i in range(num_properties):
    random_single_fault_masks[i, i] = 0
    random_single_fault_masks_with_perfect[i,i] = 0

# 1. Have the card distribution for train set be the same as test set.
# That is, we essentially pick a random faulty agent to generate set/no_set for each sequence,
# But then each sequence is evaluated by a consistent faulty agent
train_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_train, length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect)
test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, perfect_mask, random_single_fault_masks_with_perfect)

# 2. Have the card distributions be different, where its picked to be roughly 50% correct answers for the faulty agent. This is probably even more challenging for the AI
train_sequences = game.generate_faulty_check_sequences(num_sequences_train, length_sequence, random_single_fault_masks)
test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, perfect_mask, random_single_fault_masks_with_perfect)

