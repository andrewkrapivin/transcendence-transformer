import torch
import math
import random

from einops import rearrange, repeat

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
    def shuffle_cards(self, num_cards):
        assert(num_cards <= self.num_cards)
        num_cards_for_unique = 10 * num_cards * math.ceil(math.log2(num_cards)) + 10
        probs = torch.ones(self.num_values)
        self.cards = torch.multinomial(probs, num_cards_for_unique*self.num_properties, replacement=True)
        self.cards = rearrange(self.cards, "(n p) -> n p", p = self.num_properties)
        self.cards = torch.unique(self.cards, dim = 0)
        self.cards = self.cards[torch.randperm(self.cards.shape[0])]
        self.cards = self.cards[:num_cards]
        # print(self.cards)

        # probs = torch.ones((self.num_properties, self.num_values))
        # self.cards = torch.multinomial(probs, num_cards, replacement=False)

        return self.cards
    
    def check_set(self, cards):
        return self.faulty_check_set(cards, torch.ones(self.num_properties))

    def faulty_check_set(self, cards, blind_property_mask):
        assert (cards.shape[0] == self.num_values)

        # print("checking faulty cards")
        # print("passed in cards", cards)
        # print("passed in blind property mask", blind_property_mask)
        # mask off the properties that we are blind to; they will automatically be counted as correct
        cards = cards * repeat(blind_property_mask, "p -> n p", n = cards.shape[0])
        # print("with blind mask", cards)
        
        # ok unique is a little messed up so not sure at all how to do this in a proper tensor way
        # _, num_distinct_values = torch.unique(cards, dim=1)
        # print("distinct value count", num_distinct_values)
        # property_validity = (num_distinct_values == 1) + (num_distinct_values == self.num_values)
        # print("property validity", property_validity)

        # is_set = torch.logical_and(property_validity)
        # print("is set", is_set)
        
        is_set = True

        for i in range(self.num_properties):
            unique_vals = torch.unique(cards[:, i])
            # print("unique vals for property", i, unique_vals)
            is_set = is_set and (unique_vals.shape[0] == 1 or unique_vals.shape[0] == self.num_values)

        # print("is set", is_set)
        return is_set
    
    def gen_random_set_faulty(self, blind_property_mask):
        permsize = math.factorial(self.num_values)
        same_mask = (torch.randint(0, permsize+1, (self.num_properties, )) == 0).int() * blind_property_mask
        if torch.sum(same_mask) == self.num_properties:
            return self.gen_random_set()
        same_mask = repeat(same_mask, "p -> n p", n = self.num_values)
        blind_property_mask = repeat(blind_property_mask, "p -> n p", n = self.num_values)
        diff_mask = (1-same_mask) * blind_property_mask
        arb_mask = (1-blind_property_mask)
        
        probs = torch.ones(self.num_properties, self.num_values)
        cards_arb = torch.multinomial(probs, self.num_values, replacement=True)
        cards_arb = rearrange(cards_arb, "p n -> n p", p = self.num_properties)
        cards_diff = torch.multinomial(probs, self.num_values, replacement=False)
        cards_diff = rearrange(cards_diff, "p n -> n p", p = self.num_properties)
        cards_one_val = torch.multinomial(torch.ones(self.num_values), self.num_properties, replacement=True)
        cards_one_val = repeat(cards_one_val, "p -> n p", n = self.num_values)

        # print("generating random set")
        # print("all different potential cards", cards)
        # print("all same potential cards", cards_one_val)
        # print("mask", same_mask)

        # cards = cards * (1-same_mask.int()) + cards_one_val * same_mask.int()
        cards = cards_arb * arb_mask + cards_diff * diff_mask + cards_one_val * same_mask

        # print("generated cards", cards)

        return cards
    
    def gen_random_set(self):
        return self.gen_random_set_faulty(torch.ones(self.num_properties))

    def gen_random_non_set_faulty(self, blind_property_mask):
        cards = self.shuffle_cards(self.num_values)
        if self.faulty_check_set(cards, blind_property_mask):
            return self.gen_random_non_set()
        return cards
    
    def gen_random_non_set(self):
        return self.gen_random_non_set_faulty(torch.ones(self.num_properties))
    
    def find_in_set_faulty(self, cards, blind_property_mask):
        possible_sets = torch.combinations(torch.arange(cards.shape[0]), self.num_values)
        valids = []
        for i in range(possible_sets.shape[0]):
            possible_set = cards[possible_sets[i]]
            # print(possible_set)
            valid = self.faulty_check_set(possible_set, blind_property_mask)
            if valid:
                valids.append(i)

        if len(valids) == 0:
            return False, None
        return True, cards[possible_sets[valids[random.randrange(0, len(valids))]]]

    def find_in_set(self, cards):
        return self.find_in_set_faulty(cards, torch.ones(self.num_properties))

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
    def generate_faulty_check_sequences(self, num, length, possible_property_masks, set_probability = 0.5, property_mask_probabilities = None):
        if property_mask_probabilities == None:
            property_mask_probabilities = torch.ones(possible_property_masks.shape[0])
        
        property_mask_selection_idx = torch.multinomial(property_mask_probabilities, num, replacement=True)
        property_masks = possible_property_masks[property_mask_selection_idx]
        set_or_not = torch.bernoulli(torch.ones((num, length)) * set_probability).bool()

        per_card_tokens = self.num_properties + 1
        num_cards = self.num_values
        tokens_per_sentence = per_card_tokens * num_cards + 3
        sequences = torch.empty((num, length * tokens_per_sentence), dtype=torch.int)
        for i in range(num):
            offset = 0
            for j in range(length):
                sequences[i,offset] = self.SOS_token
                if set_or_not[i, j]:
                    cards = self.gen_random_set_faulty(property_masks[i])
                    answer_token = self.SET_token
                else:
                    cards = self.gen_random_non_set_faulty(property_masks[i])
                    answer_token = self.NO_SET_token
                cards_seq = self.cards_to_seq(cards)
                card_start = offset+1
                card_end = card_start + num_cards*per_card_tokens
                sequences[i, card_start: card_end] = cards_seq
                sequences[i, card_end] = answer_token
                sequences[i, card_end+1] = self.EOS_token
                offset += tokens_per_sentence
        return sequences
                
    
    def generate_perfect_check_sequences(self, num, length, set_probability = 0.5):
        return self.generate_faulty_check_sequences(num, length, torch.ones(1, self.num_properties))

    # def generate_perfect_then_faulty(self, num, length, possible_property_masks, set_probability = 0.5, property_mask_probabilities = None):
    def generate_faulty_one_way_check_faulty_another(self, num, length, checker_possible_property_masks, generator_possible_property_masks, set_probability = 0.5, checker_property_mask_probabilities = None, generator_property_mask_probabilities = None):
        if checker_property_mask_probabilities == None:
            checker_property_mask_probabilities = torch.ones(checker_possible_property_masks.shape[0])
        if generator_property_mask_probabilities == None:
            generator_property_mask_probabilities = torch.ones(generator_possible_property_masks.shape[0])
        
        checker_property_mask_selection_idx = torch.multinomial(checker_property_mask_probabilities, num, replacement=True)
        checker_property_masks = checker_possible_property_masks[checker_property_mask_selection_idx]

        generator_property_mask_selection_idx = torch.multinomial(generator_property_mask_probabilities, num*length, replacement=True)
        generator_property_masks = generator_possible_property_masks[generator_property_mask_selection_idx]
        generator_property_masks = rearrange(generator_property_masks, "(n l) p -> n l p", n = num)

        set_or_not = torch.bernoulli(torch.ones((num, length)) * set_probability).bool()

        per_card_tokens = self.num_properties + 1
        num_cards = self.num_values
        tokens_per_sentence = per_card_tokens * num_cards + 3
        sequences = torch.empty((num, length * tokens_per_sentence), dtype=torch.int)
        for i in range(num):
            offset = 0
            for j in range(length):
                sequences[i,offset] = self.SOS_token
                if set_or_not[i, j]:
                    cards = self.gen_random_set_faulty(generator_property_masks[i, j])
                    # answer_token = self.SET_token
                else:
                    cards = self.gen_random_non_set_faulty(generator_property_masks[i, j])
                    # answer_token = self.NO_SET_token
                if self.faulty_check_set(cards, checker_property_masks[i]):
                    answer_token = self.SET_token
                else:
                    answer_token = self.NO_SET_token
                cards_seq = self.cards_to_seq(cards)
                card_start = offset+1
                card_end = card_start + num_cards*per_card_tokens
                sequences[i, card_start: card_end] = cards_seq
                sequences[i, card_end] = answer_token
                sequences[i, card_end+1] = self.EOS_token
                offset += tokens_per_sentence
        return sequences

    
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