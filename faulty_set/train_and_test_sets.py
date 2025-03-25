import torch
import equal_diff_set
import sys
import os
# We probably want to vary some of these parameters
num_properties = 4
num_values = 3
# This is the number of potential sets evaluated by an agent in a sequence
train_length_sequence = 30
test_length_sequence = 30
num_sequences_train = 20000
num_sequences_test = 2000

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
# train_sequences, val_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_train, train_length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect, set_probability=0.8, val=True)
# test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, test_length_sequence, perfect_mask, random_single_fault_masks_with_perfect, set_probability=0.8)

# save_dir = "./transcendenceGPT/data/card_set_30_train_30_test/"

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# torch.save(train_sequences, save_dir + "train.pt")
# torch.save(val_sequences, save_dir + "val.pt")
# torch.save(test_sequences, save_dir + "test.pt")

# 2. Have the card distributions be different, where its picked to be roughly 50% correct answers for the faulty agent. This is probably even more challenging for the AI
# train_sequences, val_sequences = game.generate_faulty_check_sequences(num_sequences_train,train_length_sequence, random_single_fault_masks, val = True)
# test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, test_length_sequence, perfect_mask, random_single_fault_masks_with_perfect, set_probability=0.8)

# save_dir = "./transcendenceGPT/data/card_set_ood_1_train_1_test/"

# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
    
# torch.save(train_sequences, save_dir + "train.pt")
# torch.save(val_sequences, save_dir + "val.pt")
# torch.save(test_sequences, save_dir + "test.pt")

# num_sequences_test = 8000

# for lengthy_length in range(1, 31):
#     test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, lengthy_length, perfect_mask, random_single_fault_masks_with_perfect, set_probability=1)

#     torch.save(test_sequences, "./transcendenceGPT/data/in_context_learning_test/" + "len_" + str(lengthy_length) + "_test.pt")


# Uncomment to generate test data for only one expert

num_sequences_test = 1
test_length_sequence = 8000

test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, test_length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect, set_probability=0.8)
torch.save(test_sequences, "./transcendenceGPT/data/faulty_one_expert_test/test.pt")