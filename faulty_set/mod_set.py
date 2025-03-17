import torch
import torch.nn.functional as F
import math
import random

from einops import rearrange, repeat

# since we're already not exactly following set rules here, we'll make it fast

class set_game():
    def __init__(self, num_properties, num_values):
        self.num_basic_tokens = 5
        # self.end_of_property_tokens = self.num_basic_tokens + num_properties
        # self.end_of_value_tokens = self.end_of_property_tokens + num_values
        self.num_properties = num_properties
        self.num_values = num_values

        self.SOS_token = torch.tensor([0])
        self.EOS_token = torch.tensor([1])
        # card separator token
        self.CS_token = torch.tensor([2])
        self.SET_token = torch.tensor([3])
        self.NO_SET_token = torch.tensor([4])

        # maybe delete the property tokens actually? We will just always have the properties in the same order?
        # self.property_tokens = torch.arange(self.num_basic_tokens, self.end_of_property_tokens)
        # self.value_tokens = torch.arange(self.end_of_property_tokens, self.end_of_value_tokens)

        self.num_value_tokens = num_values * num_properties
        self.value_tokens = torch.arange(self.num_basic_tokens, self.num_value_tokens + self.num_basic_tokens)

        self.num_cards = num_values ** num_properties

        self.tokens_to_string = [
            " ",
            ".",
            ";",
            "S",
            "N",
        ]
        for i in range(num_properties):
            for j in range(num_values):
                self.tokens_to_string.append(chr(i + ord('a'))+str(j))
    
    # this is cursed how do we get it to not repeat stuff
    def shuffle_cards(self, num_sequences, num_groups, num_cards):
        probs = torch.ones(self.num_values)
        self.cards = torch.multinomial(probs, num_sequences*num_groups*num_cards*self.num_properties, replacement=True)
        self.cards = rearrange(self.cards, "(s g c p) -> s g c p", p = self.num_properties, g = num_groups, c = num_cards)

        return self.cards
    
    def check_set(self, cards):
        return self.faulty_check_set(cards, torch.ones(self.num_properties))


    # blind property mask can be uniformly applied to everything
    # or it can be a tensor of shape one to only apply to sequences
    # or it can be of shape 2 to apply to each group

    def maybe_expand_mask(self, mask, n, g):
        if len(mask.shape) == 1:
            mask = repeat(mask, "p -> n g p", n = n, g = g)
        if len(mask.shape) == 2:
            mask = repeat(mask, "n p -> n g p", g = g)
        return mask

    def faulty_check_set(self, cards, blind_property_mask):
        assert(cards.shape[2] == self.num_values and cards.shape[3] == self.num_properties)

        blind_property_mask = self.maybe_expand_mask(blind_property_mask, cards.shape[0], cards.shape[1])
        # if len(blind_property_mask.shape) == 1:
        #     blind_property_mask = repeat(blind_property_mask, "p -> n g p", n = cards.shape[0], g = cards.shape[1])
        # if len(blind_property_mask.shape) == 2:
        #     blind_property_mask = repeat(blind_property_mask, "n p -> n g p", g = cards.shape[1])

        value_sum = torch.sum(cards, dim=2)
        values_correct = torch.remainder(value_sum, self.num_values) == 0
        values_correct_faulty = values_correct.int() * (1-blind_property_mask)

        correct_faulty = torch.all(values_correct_faulty, dim = 2)
        
        return correct_faulty
    
    def gen_random_sets_faulty(self, num_sequences, num_groups, blind_property_mask, correct_mask):
        probs = torch.ones(self.num_values)
        # generating all but one card since the last is determined by the residue
        cards = torch.multinomial(probs, num_sequences*num_groups*(self.num_values-1)*self.num_properties, replacement=True)
        cards = rearrange(cards, "(s g c p) -> s g c p", p = self.num_properties, g = num_groups, c = self.num_values-1)

        blind_property_mask = self.maybe_expand_mask(blind_property_mask, cards.shape[0], cards.shape[1])
        correct_mask = self.maybe_expand_mask(correct_mask, cards.shape[0], cards.shape[1])

        cards_sum_mod = torch.remainder(torch.sum(cards, dim=2), self.num_values)
        completion_correct = torch.remainder(self.num_values - cards_sum_mod, self.num_values)
        cards_correct_onehot = F.one_hot(completion_correct, num_classes=self.num_values)
        cards_incorrect_probs = 1-cards_correct_onehot

        completion_random = torch.multinomial(probs, num_sequences*num_groups*self.num_properties)
        completion_random = rearrange(completion_random, "(s g c p) -> s g c p", p = self.num_properties, g = num_groups, c = 1)

        completion_incorrect = torch.multinomial(rearrange(cards_incorrect_probs, "s g p -> (s g) p"), 1, replacement=True)
        completion_incorrect = rearrange(completion_incorrect, "(s g) c-> s (g c)", s = num_sequences, c = 1)
        
        completion = (1-blind_property_mask) * completion_random + blind_property_mask * (correct_mask * completion_correct + (1-correct_mask) * completion_incorrect)

        cards = torch.cat([cards, torch.unsqueeze(cards_correct_completion, dim = 3)])

        return cards
    
    def gen_random_sets(self, num_sequences, num_groups, blind_property_mask, correct_mask):
        return self.gen_random_set_faulty(torch.ones(self.num_properties))

    def cards_to_seq(self, cards):
        assert(cards.shape[1] == self.num_properties)
        per_card_tokens = self.num_properties + 1
        property_offsets = torch.arange(0, self.num_properties) * self.num_values
        # print("cards to seq", cards, repeat(property_offsets, "p -> v p", v=self.num_values))
        cards = cards + repeat(property_offsets, "p -> v p", v=self.num_values)
        card_tokens = self.value_tokens[cards.int()]

        tokens_with_separators = torch.empty((cards.shape[0], self.num_properties+1), dtype=torch.int)
        tokens_with_separators[:, :self.num_properties] = card_tokens
        tokens_with_separators[:, self.num_properties] = torch.ones((cards.shape[0], )).int() * self.CS_token
        # print("tokens with separators", tokens_with_separators)
        return rearrange(tokens_with_separators, "v p -> (v p)")
    
    def seq_to_string(self, seq):
        # print(seq)
        return ''.join([self.tokens_to_string[t] for t in seq.tolist()])


    # Sequence is SOS C1;C2;C3;{SET or NO_SET} EOS, multiplied by #?
    # Therefore, each "sentence" has 3 + (num_properties + 1) * num_values elements
    def generate_faulty_check_sequences(self, num_sequences, num_groups, possible_property_masks, set_probability = 0.5, property_mask_probabilities = None):
        if property_mask_probabilities == None:
            property_mask_probabilities = torch.ones(possible_property_masks.shape[0])
        
        correct_masks = torch.multinomial(torch.tensor([set_probability, 1-set_probability]), num_sequences * num_groups, replacement=True)
        correct_masks = rearrange(correct_masks, "(s g) -> s g", s = num_sequences)
        property_idx = torch.multinomial(property_mask_probabilities, num_sequences * num_groups, replacement=True)
        property_masks = possible_property_masks[property_idx]
        property_masks = rearrange(property_masks, "(s g) -> s g", s = num_sequences)

        sets = self.gen_random_sets_faulty(num_sequences, num_groups, property_masks, correct_masks)

        # TODO: then need to finish this code
    
    def generate_perfect_check_sequences(self, num_sequences, num_groups, set_probability = 0.5):
        return self.generate_faulty_check_sequences(num, length, torch.ones(1, self.num_properties))

    # def generate_perfect_then_faulty(self, num, length, possible_property_masks, set_probability = 0.5, property_mask_probabilities = None):
    def generate_faulty_one_way_check_faulty_another(self, num, length, checker_possible_property_masks, generator_possible_property_masks, set_probability = 0.5, checker_property_mask_probabilities = None, generator_property_mask_probabilities = None):
        if checker_property_mask_probabilities == None:
            checker_property_mask_probabilities = torch.ones(checker_possible_property_masks.shape[0])
        if generator_property_mask_probabilities == None:
            generator_property_mask_probabilities = torch.ones(generator_possible_property_masks.shape[0])
        
        # TODO: finish

    
    # Possible TODO: Do something with the find problem, of finding a set in a group of cards
    # Seems difficult to do, as if there are multiple options then how could it learn them
    # Might still be useful to sample a random one
    # It would at least (assuming it learns) learn how to classify a group of cards as containing a set or not


if __name__ == "__main__":
    game = set_game(4, 3)
    cards = game.shuffle_cards(3)
    print(cards)
    game.check_set(cards)
    assert(game.check_set(torch.tensor([[0,0,0,0], [1,1,0,0], [2,2,0,0]])) == True)
    assert(game.check_set(torch.tensor([[0,0,0,0], [1,1,0,0], [2,2,0,1]])) == False)
    
    game = set_game(4, 4)
    # for i in range(1000):
    #     assert(game.check_set(game.gen_random_set()) == True)
    # # print("fafa")

    # for i in range(1000):
    #     assert(game.check_set(game.gen_random_non_set()) == False)

    # portion_valid = 0
    # num_set_checks = 1000
    # for i in range(num_set_checks):
    #     cards = game.shuffle_cards(18)
    #     has_set, set_cards = game.find_in_set(cards)
    #     if has_set:
    #         portion_valid += 1
    #         assert(game.check_set(set_cards) == True)
    # print("portion valid sets", portion_valid / num_set_checks)

    seqs = game.generate_perfect_check_sequences(5, 5)
    print(game.seq_to_string(seqs[2]))

    first_faulty_mask = torch.ones((1, game.num_properties))
    first_faulty_mask[0, 0] = 0
    seqs_first_faulty = game.generate_faulty_check_sequences(5, 5, first_faulty_mask)
    print(game.seq_to_string(seqs_first_faulty[2]))

    first_faulty_mask = torch.ones((2, game.num_properties))
    first_faulty_mask[0, 0] = 0
    first_faulty_mask[1, 1] = 0
    seqs_first_faulty = game.generate_faulty_check_sequences(5, 5, first_faulty_mask)
    print("0", game.seq_to_string(seqs_first_faulty[0]))
    print("1", game.seq_to_string(seqs_first_faulty[1]))
    print("2", game.seq_to_string(seqs_first_faulty[2]))
    print("3", game.seq_to_string(seqs_first_faulty[3]))
    print("4", game.seq_to_string(seqs_first_faulty[4]))

    seqs_some_faulty_diff_generator = game.generate_faulty_one_way_check_faulty_another(5, 5, first_faulty_mask, first_faulty_mask)
    print("0", game.seq_to_string(seqs_some_faulty_diff_generator[0]))
    print("1", game.seq_to_string(seqs_some_faulty_diff_generator[1]))
    print("2", game.seq_to_string(seqs_some_faulty_diff_generator[2]))
    print("3", game.seq_to_string(seqs_some_faulty_diff_generator[3]))
    print("4", game.seq_to_string(seqs_some_faulty_diff_generator[4]))