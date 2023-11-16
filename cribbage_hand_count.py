"""
Count points in a hand for cribbage. 
"""
from itertools import combinations
import numpy as np


class Card:
    """
    define card from a face and a suit. Input face a number for number cards and JQK for others.
    Face_value: the number on the card for number cards, 11 12 13 for J Q K
    Value: number on the card for number cards, 10 for J Q K
    Suit: the suit of the card
    """
    def __init__(self,face,suit):
        self.face = face
        self.value = 0
        self.face_value = 0
        self.suit = suit

        faces = ['J','Q','K']
        face_faces = {'J':11,'Q':12,'K':13}

        if self.face in faces:
            self.value = 10
        else:
            self.value = int(self.face)

        if self.face not in faces:
            self.face_value = self.value
        else:
            self.face = face_faces[self.face]



#-----------------------------------------------
# create all possible hands
####last card is drawn card
#-----------------------------------------------


class Hand:
    """
    A hand is made up of 4 + 1 Cards.
    """
    def __init__(self,handset):
        self.cards = set()
        self.points = 0
        self.handset = list(handset)
        self.handvalues = [c.value for c in self.handset]
        self.handfaces = [c.face_value for c in self.handset]
        self.handsuits = [c.suit for c in self.handset]
        self.hfset = set(handset)


    def evaluate_strict_suits(self):
        """
        Get suits of the hand without the top card. 
        """
        return self.handsuits[:-1]

    def evaluate_points(self):
        """
        Collect all the points in the hand. 
        """
        # Fifteen
        for i in range(2,6):
            combs_length_i = combinations(self.handvalues,i)
            for comb in combs_length_i:
                if sum(list(comb)) == 15:
                    self.points +=2

        # Pairs
        for f_c in self.hfset:
            if self.handfaces.count(f_c)==2:
                self.points+=2
            elif self.handfaces.count(f_c)==3:
                self.points+=6
            elif self.handfaces.count(f_c)==4:
                self.points+=12

        # Sequence
        sorted_faces = sorted(self.handfaces)
        max_len = len(max(np.split(sorted_faces, np.where(np.diff(sorted_faces)!=1)[0]+1),key=len))
        if max_len>2:
            self.points += max_len

        # Flush
        if len(set(self.handsuits)) == 1:
            self.points += 5
        elif len(set(self.handsuits)) == 2:
            deal_suits = self.evaluate_strict_suits()
            if len(set(deal_suits)) == 1:
                self.points += 4

        # One for his nob
        top_suit = self.handsuits[-1]
        for i in range(4):
            if self.handsuits[i] == top_suit and self.handfaces[i]==11:
                self.points+=1

        return self.points


SUITS = ['H','C','S','D']


DECK = [Card(i,s) for i in range(1,11) for s in SUITS]
DECK += [Card(i,s) for i in ['J','Q','K'] for s in SUITS]

H4 = combinations(DECK,4)

def hand_generator():
    """
    Use generator to make this faster.
    Can be used to extract list if so desired.
    """
    for h_h in H4:
        top_cards = [t for t in DECK if t not in h_h]
        for t_c in top_cards:
            yield h_h+(t_c,)

# hand_points = []
# for hand in hand_generator():
#     hand_points.append(hand.evaluate_points())

hand_points = [Hand(h).evaluate_points() for h in hand_generator()]

vals_dist = [hand_points.count(j) for j in range(0,30)]
