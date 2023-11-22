"""
Count points in a hand for cribbage. 
"""
from itertools import combinations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chisquare


class Card:
    """
    define card from a face and a suit. Input face a number for number cards and JQK for others.
    Face_value: the number on the card for number cards, 11 12 13 for J Q K
    Value: number on the card for number cards, 10 for J Q K
    Suit: the suit of the card
    """
    def __init__(self,face,suit):
        self.face = face
        self.suit = suit
        self.value()
        self.face_value()


    def value(self):
        faces = ['J','Q','K']
        if self.face in faces:
            return 10
        else:
            return  int(self.face)
    def face_value(self):
        faces = ['J','Q','K']
        face_faces = {'J':11,'Q':12,'K':13}
        if self.face not in faces:
            return self.value()
        else:
            return face_faces[self.face]





class Hand:
    """
    A hand is made up of 4 + 1 Cards. Last card is the drawn card.
    """
    def __init__(self,handset):
        self.cards = set()
        self.points = 0
        self.handset = handset
        self.handvalues = [c.value() for c in self.handset]
        self.handfaces = [c.face_value() for c in self.handset]
        self.handsuits = [c.suit for c in self.handset]


    def evaluate_strict_suits(self):
        """
        Get suits of the hand without the top card. 
        """
        return self.handsuits[:-1]

    def evaluate_points(self):
        """
        Collect all the points in the hand. 
        """
        p=0
        # Fifteen
        for i in range(2,6):
            combs_length_i = combinations(self.handvalues,i)
            for comb in combs_length_i:
                if sum(list(comb)) == 15:
                    p+=2# self.points +=2

        # Pairs
        for f_c in set(self.handfaces):
            if self.handfaces.count(f_c)==2:
                # print('2pair')
                p+=2#self.points+=2
            elif self.handfaces.count(f_c)==3:
                # print('3pair')
                p+=6#self.points+=6
            elif self.handfaces.count(f_c)==4:
                # print('4pair')
                p+=12#self.points+=12

        # Sequence
        sorted_faces = sorted(self.handfaces)
        max_len = len(max(np.split(sorted_faces, np.where(np.diff(sorted_faces)!=1)[0]+1),key=len))
        if max_len>2:
            p+=max_len#self.points += max_len
            # print('seq')

        # Flush
        if len(set(self.handsuits)) == 1:
            p+=5
        elif len(set(self.handsuits)) == 2:
            deal_suits = self.evaluate_strict_suits()
            if len(set(deal_suits)) == 1:
                # print('flush')
                p+=4

        # One for his nob
        top_suit = self.handsuits[-1]
        for i in range(4):
            if self.handsuits[i] == top_suit and self.handfaces[i]==11:
                p+=1#self.points+=1

        return p#self.points


if __name__=='__main__':
        

    SUITS = ['H','C','S','D']


    DECK = [Card(i,s) for i in range(1,11) for s in SUITS]
    DECK += [Card(i,s) for i in ['J','Q','K'] for s in SUITS]

    H4 = combinations(DECK,4)

    def all_hands():
        """
        Use generator to make this faster.
        Can be used to extract list if so desired.
        """
        for h_h in H4:
            top_cards = [t for t in DECK if t not in h_h]
            for t_c in top_cards:
                yield h_h+(t_c,)

    hand_points = [Hand(h).evaluate_points() for h in all_hands()]

    # Plots

    vals_dist = [hand_points.count(j)/len(hand_points) for j in range(0,30)]
    vals_dist_2130 = [hand_points.count(j)/len(hand_points) for j in range(21,30)]

    df = pd.DataFrame({'dist':vals_dist_2130,'score': [j for j in range(21,30)]} )



    sns.set_theme(style="darkgrid")
    sns.catplot(data = df,x='score',y='dist', kind='bar',color=sns.color_palette()[2])
    plt.title('Points distribution with random choice for scores between 21 and 29')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.show()



    with open(r'hand.txt','r') as file:
        pre = file.read().split(',')

    hand = [int(val) for val in pre]

    with open(r'crib.txt','r') as file:
        pre = file.read().split(',')

    crib = [int(val) for val in pre]
    
    hand_count = [hand.count(j)/len(hand) for j in range(0,30)]
    crib_count = [crib.count(j)/len(crib) for j in range(0,30)]

    sns.set_theme(style="darkgrid")
    sns.catplot(data=hand_count, kind='bar',color=sns.color_palette()[0])
    plt.title('Hand points distribution')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.show()
    
    sns.set_theme(style="darkgrid")
    sns.catplot(data=crib_count, kind='bar',color=sns.color_palette()[1])
    plt.title('Crib points distribution')
    plt.xlabel('points')
    plt.ylabel('frequency')
    plt.show()



    conc_scores = [j for j in range(0,30)] + [j for j in range(0,30)] 
    conc_freq = hand_count + crib_count
    origin_labels = ['hand' for j in range(0,30)] + ['crib' for j in range(0,30)] 


    data_df = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})


    sns.set_theme(style="darkgrid")
    sns.catplot(data_df, x = 'score', y = 'frequency',hue='origin',legend=True , legend_out=False, kind='bar')
    plt.title('Hand and crib points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.show()


    conc_freq = hand_count + vals_dist
    origin_labels = ['hand' for j in range(0,30)] +['random choice' for j in range(0,30)]


    df_handtotal = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})

    sns.set_theme(style="darkgrid")
    sns.catplot(df_handtotal, x = 'score', y = 'frequency',hue='origin', legend=True , legend_out=False, kind='bar', palette=[sns.color_palette()[0],sns.color_palette()[2]])
    plt.title('Hand and random choice points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.show()


    conc_freq = crib_count + vals_dist
    origin_labels = ['crib' for j in range(0,30)] + ['random choice' for j in range(0,30)]


    df_cribtotal = ({'score':conc_scores, 'origin': origin_labels, 'frequency': conc_freq})
   
    sns.set_theme(style="darkgrid")
    sns.catplot(df_cribtotal, x = 'score', y = 'frequency',hue='origin',legend=True , legend_out=False, kind='bar', palette=[sns.color_palette()[1],sns.color_palette()[2]])
    plt.title('Crib and random choice points distribution')
    plt.xlabel('points')
    plt.legend(loc="upper right")
    plt.ylabel('frequency')
    plt.show()


    # Statistics
    hand_full = [hand.count(j) for j in range(0,30)]
    crib_full = [crib.count(j) for j in range(0,30)]
    rand_full_hand = [hand_points.count(j)*(len(hand))/len(hand_points) for j in range(0,30)]
    rand_full_crib = [hand_points.count(j)*(len(crib))/len(hand_points) for j in range(0,30)]

    hand_mean = np.mean(hand)
    crib_mean = np.mean(crib)
    total_mean = np.mean(hand_points)

    hand_var = np.var(hand)
    crib_var = np.var(crib)
    total_var = np.var(hand_points)

    df_chi = pd.DataFrame({'random_hand':rand_full_hand,'random_crib':rand_full_crib,'hand':hand_full,'crib':crib_full})

    df_chi = df_chi.drop(df_chi[df_chi['random_hand']==0].index)


    chi_hand = chisquare(f_obs=df_chi['hand'], f_exp=df_chi['random_hand'], ddof=len(hand)-1)
    chi_crib = chisquare(f_obs=df_chi['crib'], f_exp=df_chi['random_crib'], ddof=len(crib)-1)

    print(hand_mean,crib_mean,total_mean)
    print(hand_var,crib_var,total_var)
    print(chi_hand)
    print(chi_crib)
    