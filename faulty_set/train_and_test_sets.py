import torch
import equal_diff_set
import sys
# We probably want to vary some of these parameters
num_properties = 4
num_values = 3
# This is the number of potential sets evaluated by an agent in a sequence
length_sequence = 50
num_sequences_train = 60000
num_sequences_test = 1000

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
# train_sequences, val_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_train, length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect, val=True)
# test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, perfect_mask, random_single_fault_masks_with_perfect)

# # print(train_sequences.shape)
# # sys.exit(1)
# torch.save(train_sequences, "./transcendenceGPT/data/card_set/train.pt")
# torch.save(val_sequences, "./transcendenceGPT/data/card_set/val.pt")
# torch.save(test_sequences, "./transcendenceGPT/data/card_set/test.pt")

# # 2. Have the card distributions be different, where its picked to be roughly 50% correct answers for the faulty agent. This is probably even more challenging for the AI
# train_sequences = game.generate_faulty_check_sequences(num_sequences_train, length_sequence, random_single_fault_masks)
# test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, perfect_mask, random_single_fault_masks_with_perfect)

# torch.save(train_sequences, "./transcendenceGPT/data/card_set2/train.pt")
# torch.save(val_sequences, "./transcendenceGPT/data/card_set2/val.pt")
# torch.save(test_sequences, "./transcendenceGPT/data/card_set2/test.pt")

train_sequences, val_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_train, length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect, val=True, set_probability = 0.8)
test_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, perfect_mask, random_single_fault_masks_with_perfect, set_probability = 0.8)
test_in_dist_sequences = game.generate_faulty_one_way_check_faulty_another(num_sequences_test, length_sequence, random_single_fault_masks, random_single_fault_masks_with_perfect, set_probability = 0.8)

# print("train", train_sequences.shape)
# print("val", val_sequences.shape)
# print("test", test_sequences.shape)
torch.save(train_sequences, "./transcendenceGPT/data/card_set_50x50_big/train.pt")
torch.save(val_sequences, "./transcendenceGPT/data/card_set_50x50_big/val.pt")
torch.save(test_sequences, "./transcendenceGPT/data/card_set_50x50_big/test.pt")
torch.save(test_in_dist_sequences, "./transcendenceGPT/data/card_set_50x50_big/test_in_dist.pt")