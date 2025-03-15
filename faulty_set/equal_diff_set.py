import torch
import math
import random

from einops import rearrange, repeat

class set_game():
    def __init__(self, num_properties, num_values):
        self.end_of_basic_tokens = 4
        # self.end_of_property_tokens = self.end_of_basic_tokens + num_properties
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
        # self.property_tokens = torch.arange(self.end_of_basic_tokens, self.end_of_property_tokens)
        # self.value_tokens = torch.arange(self.end_of_property_tokens, self.end_of_value_tokens)

        self.end_of_value_tokens = self.end_of_basic_tokens + num_values * num_properties
        self.value_tokens = torch.arange(self.end_of_basic_tokens, self.end_of_value_tokens)

        self.num_cards = num_values ** num_properties
    
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
        cards_with_separators = torch.empty((cards.shape[0], self.num_properties+1))
        cards_with_separators[:, :self.num_properties] = cards
        cards_with_separators[:, self.num_properties] = torch.ones((cards.shape[0], )) * self.CS_token
        property_offsets = torch.arange(0, self.num_properties) * self.num_values
        cards_with_separators = cards_with_separators + repeat(property_offsets, "p -> v p", v=self.num_values)
        tokens = self.value_tokens[cards_with_separators]
        return rearrange(tokens, "v p -> (v p)")
    
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
        sequences = torch.empty((num, length * tokens_per_sentence))
        for i in range(num):
            for j in range(length):
                if set_or_not[num, length]:
                    cards = self.gen_random_set_faulty(property_masks[i])
                else:
                    cards = self.gen_random_non_set_faulty(property_masks[i])
                
    
    def generate_perfect_check_sequences(self, num, length):
        return sefl.generate_faulty_check_sequences(num, length, torch.ones(1, self.num_properties))


if __name__ == "__main__":
    game = set_game(4, 3)
    cards = game.shuffle_cards(3)
    print(cards)
    game.check_set(cards)
    assert(game.check_set(torch.tensor([[0,0,0,0], [1,1,0,0], [2,2,0,0]])) == True)
    assert(game.check_set(torch.tensor([[0,0,0,0], [1,1,0,0], [2,2,0,1]])) == False)

    for i in range(1000):
        assert(game.check_set(game.gen_random_set()) == True)
    # print("fafa")

    for i in range(1000):
        assert(game.check_set(game.gen_random_non_set()) == False)

    portion_valid = 0
    num_set_checks = 1000
    for i in range(num_set_checks):
        cards = game.shuffle_cards(8)
        has_set, set_cards = game.find_in_set(cards)
        if has_set:
            portion_valid += 1
            assert(game.check_set(set_cards) == True)
    print("portion valid sets", portion_valid / num_set_checks)